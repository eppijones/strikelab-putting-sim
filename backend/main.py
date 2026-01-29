"""
StrikeLab Putting Sim - Main entry point.
FastAPI server with WebSocket streaming and camera capture loop.
"""

import argparse
import asyncio
import cv2
import json
import logging
import time
import threading
from pathlib import Path
from typing import Optional, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

from .camera import Camera, CameraMode, FrameData
from .detector import BallDetector, Detection
from .tracker import BallTracker, TrackerState, ShotState, VirtualBallState
from .calibration import Calibrator, AutoCalibrator
from .predictor import BallPredictor
from .config import get_config_manager, get_config
from .lens_calibration import load_lens_calibration, get_undistort_maps, LENS_PARAMS_FILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FPSMetrics:
    """Frame rate metrics."""
    cap_fps: float = 0.0   # Camera capture FPS
    proc_fps: float = 0.0  # Processing FPS
    disp_fps: float = 0.0  # Display/WebSocket FPS
    proc_latency_ms: float = 0.0


class FPSTracker:
    """Track FPS using sliding window of timestamps."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: list[float] = []
    
    def tick(self) -> float:
        """Record timestamp and return current FPS."""
        now = time.time()
        self.timestamps.append(now)
        
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        if len(self.timestamps) < 2:
            return 0.0
        
        dt = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / dt if dt > 0 else 0.0


class PuttingSimApp:
    """Main application coordinating all components."""
    
    def __init__(
        self,
        camera_mode: CameraMode = CameraMode.ARDUCAM,
        replay_path: Optional[str] = None
    ):
        self.camera_mode = camera_mode
        self.replay_path = replay_path
        
        # Components - load config first
        config = get_config()
        
        self.camera: Optional[Camera] = None
        
        # Initialize detector with config values
        self.detector = BallDetector(
            white_lower=np.array([
                config.detector.white_lower_h,
                config.detector.white_lower_s,
                config.detector.white_lower_v
            ]),
            white_upper=np.array([
                config.detector.white_upper_h,
                config.detector.white_upper_s,
                config.detector.white_upper_v
            ]),
            min_radius=config.detector.min_radius,
            max_radius=config.detector.max_radius,
            min_area=config.detector.min_area,
            max_area=config.detector.max_area,
            min_circularity=config.detector.min_circularity
        )
        
        # Initialize tracker with config values
        self.tracker = BallTracker(
            detector=self.detector,
            motion_threshold_px=config.tracker.motion_threshold_px,
            motion_confirm_frames=config.tracker.motion_confirm_frames,
            stopped_velocity_threshold=config.tracker.stopped_velocity_threshold,
            stopped_confirm_frames=config.tracker.stopped_confirm_frames,
            cooldown_duration_ms=config.tracker.cooldown_duration_ms,
            idle_ema_alpha=config.tracker.idle_ema_alpha,
            valid_motion_angle_deg=getattr(config.tracker, 'valid_motion_angle_deg', 45.0),
            forward_direction_deg=getattr(config.tracker, 'forward_direction_deg', 0.0)
        )
        
        # Set deceleration from config (in m/s², will be converted to px/s² using current calibration)
        decel_m_s2 = getattr(config.calibration, 'virtual_deceleration_m_s2', 0.55)
        self.tracker.set_deceleration_m_s2(decel_m_s2)
        
        # Sync forward direction from calibration to tracker (calibration is the source of truth)
        if config.calibration.is_valid():
            self.tracker.set_forward_direction(config.calibration.forward_direction_deg)
        
        self.calibrator = Calibrator()
        self.auto_calibrator = AutoCalibrator(stabilization_frames=30)
        self.predictor = BallPredictor(
            friction_coefficient=config.prediction.friction_coefficient,
            min_velocity_threshold=config.prediction.min_velocity_threshold,
            max_prediction_time_s=config.prediction.max_prediction_time_s
        )
        
        # State
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._current_state: Optional[TrackerState] = None
        self._current_frame: Optional[np.ndarray] = None
        self._frame_id = 0
        
        # Locked calibration for completed shots (prevents distance changing after STOPPED)
        self._shot_locked_ppm: Optional[float] = None
        self._last_shot_state: Optional[ShotState] = None
        
        # FPS tracking
        self._cap_fps = FPSTracker()
        self._proc_fps = FPSTracker()
        self._disp_fps = FPSTracker()
        self._last_proc_time = 0.0
        
        # WebSocket clients
        self._ws_clients: Set[WebSocket] = set()
        self._ws_lock = asyncio.Lock()
        
        # Load calibration from config (config already loaded above)
        if config.calibration.is_valid():
            self.calibrator.load_from_config({
                "homography_matrix": config.calibration.homography_matrix,
                "pixels_per_meter": config.calibration.pixels_per_meter,
                "origin_px": config.calibration.origin_px,
                "forward_direction_deg": config.calibration.forward_direction_deg
            })
        
        # Load lens distortion calibration (one-time setup)
        self._lens_calibrated = False
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._undistort_map1: Optional[np.ndarray] = None
        self._undistort_map2: Optional[np.ndarray] = None
        
        lens_result = load_lens_calibration()
        if lens_result and lens_result.success:
            self._camera_matrix = np.array(lens_result.camera_matrix)
            self._dist_coeffs = np.array(lens_result.dist_coeffs)
            self._lens_calibrated = True
            logger.info(f"Lens calibration loaded (error: {lens_result.reprojection_error:.3f}px)")
        else:
            logger.info("No lens calibration - run 'python -m backend.lens_calibration --live' to calibrate")
    
    def start(self):
        """Start camera capture and processing."""
        if self._running:
            return
        
        logger.info(f"Starting PuttingSim with mode={self.camera_mode.value}")
        
        # Initialize camera
        self.camera = Camera(
            mode=self.camera_mode,
            replay_path=self.replay_path
        )
        
        if not self.camera.start():
            logger.error("Failed to start camera")
            return
        
        self._running = True
        
        # Start capture thread
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        logger.info("PuttingSim started")
    
    def stop(self):
        """Stop capture and cleanup."""
        self._running = False
        
        if self.camera:
            self.camera.stop()
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        
        logger.info("PuttingSim stopped")
    
    def _capture_loop(self):
        """Main capture and processing loop (runs in separate thread)."""
        while self._running:
            frame_data = self.camera.read_frame()
            
            if frame_data is None:
                if self.camera_mode == CameraMode.REPLAY:
                    logger.info("Replay finished")
                    self._running = False
                continue
            
            # Track capture FPS
            cap_fps = self._cap_fps.tick()
            
            # Process frame
            proc_start = time.time()
            self._process_frame(frame_data)
            proc_time = (time.time() - proc_start) * 1000
            
            # Track processing FPS
            proc_fps = self._proc_fps.tick()
            
            self._last_proc_time = proc_time
            self._frame_id = frame_data.frame_id
            # Store undistorted frame for video streaming
            if self._lens_calibrated and self._undistort_map1 is not None:
                self._current_frame = cv2.remap(frame_data.frame, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)
            else:
                self._current_frame = frame_data.frame
    
    def _process_frame(self, frame_data: FrameData):
        """Process a single frame."""
        frame = frame_data.frame
        
        # Apply lens undistortion if calibrated
        if self._lens_calibrated and self._camera_matrix is not None:
            # Initialize undistort maps on first frame (for faster undistortion)
            if self._undistort_map1 is None:
                h, w = frame.shape[:2]
                self._undistort_map1, self._undistort_map2 = get_undistort_maps(
                    self._camera_matrix, self._dist_coeffs, (w, h)
                )
                logger.info(f"Initialized undistortion maps for {w}x{h}")
            
            # Fast undistortion using precomputed maps
            frame = cv2.remap(frame, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)
        
        # Detect ball on undistorted frame
        detection = self.detector.detect(frame)
        
        # Update tracker with frame for background model motion detection
        self._current_state = self.tracker.update(
            detection,
            frame_data.timestamp_ns,
            frame_data.frame_id,
            frame=frame  # Use undistorted frame
        )
        
        # Determine if a shot is active (any state except ARMED)
        current_state = self._current_state.state if self._current_state else ShotState.ARMED
        is_shot_active = current_state not in (ShotState.ARMED, ShotState.COOLDOWN)
        
        # Lock calibration at TRACKING start (not STOPPED) to prevent mid-shot drift
        if current_state == ShotState.TRACKING and self._shot_locked_ppm is None:
            self._shot_locked_ppm = self.get_pixels_per_meter()
            self.tracker.set_calibration(self._shot_locked_ppm)
            logger.info(f"Calibration locked at TRACKING start: {self._shot_locked_ppm:.1f} px/m")
        elif current_state == ShotState.ARMED:
            # Reset lock when returning to ARMED (ready for new shot)
            self._shot_locked_ppm = None
        
        # Auto-calibrate from ball size only when idle (ball is stationary, no shot active)
        if detection and self._current_state:
            self.auto_calibrator.update(
                detection.radius, 
                detection.confidence, 
                is_shot_active=is_shot_active
            )
            # Update tracker with current calibration when ARMED (not during shot)
            if self.auto_calibrator.is_calibrated and current_state == ShotState.ARMED:
                self.tracker.set_calibration(self.auto_calibrator.pixels_per_meter)
    
    def get_pixels_per_meter(self) -> float:
        """
        Get the best available pixels_per_meter value.
        Priority: manual override > auto-calibration > homography calibration > default
        Applies distance_scale_factor from config for fine-tuning.
        """
        config = get_config()
        
        # Check for manual override first (most reliable if user has measured)
        manual_ppm = getattr(config.calibration, 'manual_pixels_per_meter', 0.0)
        if manual_ppm and manual_ppm > 0:
            base_ppm = manual_ppm
        elif self.auto_calibrator.is_calibrated:
            base_ppm = self.auto_calibrator.pixels_per_meter
        elif self.calibrator.is_calibrated:
            base_ppm = self.calibrator.pixels_per_meter
        else:
            # Fallback default (approximate for 80cm height, 70° FOV)
            base_ppm = 1150.0
        
        # Apply distance scale factor from config (for real-world calibration adjustment)
        # If scale_factor > 1.0, distances will be larger (compensates for underestimate)
        # Dividing ppm by scale_factor has same effect as multiplying distance by scale_factor
        scale_factor = getattr(config.calibration, 'distance_scale_factor', 1.0)
        if scale_factor and scale_factor != 1.0:
            return base_ppm / scale_factor
        return base_ppm
    
    def get_state_message(self) -> dict:
        """Build state message for WebSocket."""
        state = self._current_state
        
        # Use locked calibration if available (set at TRACKING start), otherwise get fresh value
        # This prevents distance drift during and after shots
        pixels_per_meter = self._shot_locked_ppm if self._shot_locked_ppm else self.get_pixels_per_meter()
        
        # Ball data
        ball_data = None
        ball_visible = False
        if state and state.ball_x is not None:
            ball_data = {
                "x_px": state.ball_x,
                "y_px": state.ball_y,
                "radius_px": state.ball_radius,
                "confidence": state.ball_confidence
            }
            # Check if ball is visible in frame
            resolution = self.camera.resolution if self.camera else (1280, 800)
            ball_visible = bool(
                0 <= state.ball_x <= resolution[0] and
                0 <= state.ball_y <= resolution[1]
            )
        
        # Velocity data
        velocity_data = None
        if state and state.velocity:
            velocity_data = {
                "vx_px_s": state.velocity.vx,
                "vy_px_s": state.velocity.vy,
                "speed_px_s": state.velocity.speed
            }
        
        # Virtual ball data (when ball is rolling virtually after exiting frame)
        virtual_ball_data = None
        if state and state.virtual_ball:
            vb = state.virtual_ball
            # Convert virtual distance to meters
            virtual_distance_m = vb.distance_traveled / pixels_per_meter
            
            virtual_ball_data = {
                "x": round(vb.x, 1),
                "y": round(vb.y, 1),
                "vx": round(vb.vx, 1),
                "vy": round(vb.vy, 1),
                "speed_px_s": round(vb.speed, 1),
                "speed_m_s": round(vb.speed / pixels_per_meter, 3),
                "distance_px": round(vb.distance_traveled, 1),
                "distance_m": round(virtual_distance_m, 3),
                "distance_cm": round(virtual_distance_m * 100, 1),
                "time_since_exit_s": round(vb.time_since_exit, 2),
                "is_rolling": bool(vb.is_rolling),
                "final_position": vb.final_position
            }
        
        # Exit state data (when ball has exited frame)
        exit_data = None
        if state and state.exit_state:
            es = state.exit_state
            exit_data = {
                "position": es.position,
                "velocity": es.velocity,
                "speed_px_s": round(es.speed, 1),
                "direction_deg": round(es.direction_deg, 2),
                "curvature": round(es.curvature, 4),
                "trajectory_before_exit": es.trajectory_before_exit[-50:]  # Last 50 points
            }
        
        # Prediction data (when ball exits frame or after shot)
        prediction_data = None
        if state and state.velocity and not ball_visible and state.state != ShotState.VIRTUAL_ROLLING:
            # Ball has exited - generate prediction (only if not already in virtual rolling)
            if state.ball_x is not None and state.velocity.speed > 50:
                prediction = self.predictor.predict(
                    exit_position=(state.ball_x, state.ball_y),
                    exit_velocity=(state.velocity.vx, state.velocity.vy)
                )
                if prediction:
                    prediction_data = {
                        "trajectory": [(p.x, p.y) for p in prediction.trajectory[:50]],
                        "final_position": prediction.final_position,
                        "final_time_s": round(prediction.final_time, 2),
                        "exit_speed_px_s": round(prediction.initial_speed, 1)
                    }
        
        # Shot result - use FROZEN values from the shot result to prevent jumping
        shot_data = None
        if state and state.shot_result:
            result = state.shot_result
            
            # Use the frozen pixels_per_meter from when the shot was computed
            # This prevents the distance from changing after the shot stops
            shot_ppm = result.pixels_per_meter if result.pixels_per_meter > 0 else pixels_per_meter
            
            # Convert to world coordinates using frozen calibration
            speed_m_s = result.initial_speed_px_s / shot_ppm
            direction_deg = result.initial_direction_deg
            
            # Use FROZEN distances from shot result (computed once at stop time)
            physical_distance_px = result.physical_distance_px
            physical_distance_m = physical_distance_px / shot_ppm
            
            virtual_distance_px = result.virtual_distance_px
            virtual_distance_m = virtual_distance_px / shot_ppm
            
            total_distance_px = result.total_distance_px
            total_distance_m = total_distance_px / shot_ppm
            
            shot_data = {
                "speed_m_s": round(speed_m_s, 3),
                "speed_px_s": round(result.initial_speed_px_s, 1),
                "direction_deg": round(direction_deg, 2),
                # Physical distance (visible in camera) - FROZEN
                "physical_distance_m": round(physical_distance_m, 4),
                "physical_distance_cm": round(physical_distance_m * 100, 1),
                "physical_distance_px": round(physical_distance_px, 1),
                # Virtual distance (simulated off-frame) - FROZEN
                "virtual_distance_m": round(virtual_distance_m, 4),
                "virtual_distance_cm": round(virtual_distance_m * 100, 1),
                "virtual_distance_px": round(virtual_distance_px, 1),
                # Total distance (physical + virtual) - FROZEN
                "distance_m": round(total_distance_m, 4),
                "distance_cm": round(total_distance_m * 100, 1),
                "distance_px": round(total_distance_px, 1),
                # Legacy fields for compatibility
                "frames_to_tracking": result.frames_to_tracking,
                "frames_to_speed": result.frames_to_speed,
                "duration_ms": round(result.duration_ms, 1),
                "trajectory": result.trajectory[-50:],  # Last 50 points
                "exited_frame": bool(result.exited_frame)
            }
        
        # Metrics
        metrics = {
            "cap_fps": round(self._cap_fps.tick(), 1),
            "proc_fps": round(self._proc_fps.tick(), 1),
            "disp_fps": round(self._disp_fps.tick(), 1),
            "proc_latency_ms": round(self._last_proc_time, 2),
            "idle_stddev": round(state.idle_stddev, 2) if state else 0.0
        }
        
        return {
            "frame_id": self._frame_id,
            "timestamp_ms": time.time() * 1000,
            "state": state.state.value if state else "ARMED",
            "lane": state.lane.value if state else "IDLE",
            "ball": ball_data,
            "ball_visible": ball_visible,
            "velocity": velocity_data,
            "prediction": prediction_data,
            "virtual_ball": virtual_ball_data,
            "exit_state": exit_data,
            "shot": shot_data,
            "metrics": metrics,
            "calibrated": self.calibrator.is_calibrated or self.auto_calibrator.is_calibrated,
            "auto_calibrated": self.auto_calibrator.is_calibrated,
            "lens_calibrated": self._lens_calibrated,
            "pixels_per_meter": round(pixels_per_meter, 1),
            "resolution": list(self.camera.resolution) if self.camera else [1280, 800]
        }
    
    async def broadcast_state(self):
        """Broadcast state to all WebSocket clients."""
        if not self._ws_clients:
            return
        
        message = self.get_state_message()
        message_json = json.dumps(message)
        
        async with self._ws_lock:
            disconnected = set()
            for ws in self._ws_clients:
                try:
                    await ws.send_text(message_json)
                except Exception:
                    disconnected.add(ws)
            
            self._ws_clients -= disconnected
    
    async def add_client(self, websocket: WebSocket):
        """Add WebSocket client."""
        async with self._ws_lock:
            self._ws_clients.add(websocket)
            logger.info(f"Client connected. Total: {len(self._ws_clients)}")
    
    async def remove_client(self, websocket: WebSocket):
        """Remove WebSocket client."""
        async with self._ws_lock:
            self._ws_clients.discard(websocket)
            logger.info(f"Client disconnected. Total: {len(self._ws_clients)}")


# Global app instance
app_instance: Optional[PuttingSimApp] = None


def get_app_instance() -> PuttingSimApp:
    """Get global app instance."""
    global app_instance
    if app_instance is None:
        raise RuntimeError("App not initialized")
    return app_instance


# FastAPI setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown."""
    # Startup
    global app_instance
    if app_instance:
        app_instance.start()
        
        # Start broadcast loop
        async def broadcast_loop():
            while app_instance and app_instance._running:
                await app_instance.broadcast_state()
                await asyncio.sleep(1/60)  # 60 Hz broadcast rate
        
        asyncio.create_task(broadcast_loop())
    
    yield
    
    # Shutdown
    if app_instance:
        app_instance.stop()


app = FastAPI(
    title="StrikeLab Putting Sim",
    version="1.0.0",
    lifespan=lifespan
)

# Serve static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
async def root():
    """Serve main page."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>StrikeLab Putting Sim</h1><p>Frontend not found</p>")


@app.get("/api/video")
async def video_feed():
    """MJPEG video stream for viewing camera output at ~60fps with low latency."""
    
    def generate_frames():
        import cv2
        sim = get_app_instance()
        last_frame_time = 0
        
        while sim._running:
            # Increased to ~60fps for lower perceived latency
            now = time.time()
            if now - last_frame_time < 1/60:
                time.sleep(0.002)
                continue
            last_frame_time = now
            
            if sim._current_frame is not None:
                frame = sim._current_frame.copy()
                
                # Draw ball detection overlay
                state = sim._current_state
                if state and state.ball_x is not None:
                    cx, cy = round(state.ball_x), round(state.ball_y)
                    radius = int(state.ball_radius or 15)
                    
                    # Color based on state
                    color = (0, 255, 0)  # Green for ARMED
                    if state.state.value == "TRACKING":
                        color = (0, 0, 255)  # Red for TRACKING
                    elif state.state.value == "STOPPED":
                        color = (0, 255, 255)  # Yellow for STOPPED
                    
                    # Draw ball circle and crosshair
                    cv2.circle(frame, (cx, cy), radius, color, 2)
                    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 1)
                    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 1)
                    
                    # Draw state
                    cv2.putText(frame, state.state.value, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Show speed if tracking
                    if state.velocity and state.state.value == "TRACKING":
                        speed_ms = state.velocity.speed / sim.calibrator.pixels_per_meter
                        cv2.putText(frame, f"{speed_ms:.2f} m/s", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Encode as JPEG - lower quality for faster encoding/transfer
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    try:
        sim = get_app_instance()
        return JSONResponse({
            "running": sim._running,
            "camera_mode": sim.camera_mode.value,
            "calibrated": sim.calibrator.is_calibrated,
            "state": sim.get_state_message()
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/config")
async def get_config_endpoint():
    """Get current configuration."""
    config = get_config()
    return JSONResponse({
        "camera": {
            "width": config.camera.width,
            "height": config.camera.height,
            "fps": config.camera.fps
        },
        "detector": {
            "white_lower": [config.detector.white_lower_h, config.detector.white_lower_s, config.detector.white_lower_v],
            "white_upper": [config.detector.white_upper_h, config.detector.white_upper_s, config.detector.white_upper_v]
        },
        "calibration": {
            "calibrated": config.calibration.is_valid(),
            "pixels_per_meter": config.calibration.pixels_per_meter,
            "forward_direction_deg": config.calibration.forward_direction_deg
        }
    })


@app.post("/api/calibrate/rectangle")
async def calibrate_rectangle(data: dict):
    """
    Calibrate using 4 rectangle corners.
    
    Expected data:
    {
        "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "width_m": 0.5,
        "height_m": 1.0,
        "forward_edge": "right"
    }
    """
    try:
        sim = get_app_instance()
        
        corners = [tuple(c) for c in data["corners"]]
        width_m = data["width_m"]
        height_m = data["height_m"]
        forward_edge = data.get("forward_edge", "right")
        
        result = sim.calibrator.calibrate_from_rectangle(
            corners, width_m, height_m, forward_edge
        )
        
        if result.success:
            # Save to config - convert numpy types to Python native types
            config_mgr = get_config_manager()
            config_mgr.update_calibration(
                homography_matrix=[[float(v) for v in row] for row in result.homography_matrix.tolist()],
                pixels_per_meter=float(result.pixels_per_meter),
                origin_px=(int(result.origin_px[0]), int(result.origin_px[1])),
                forward_direction_deg=float(result.forward_direction_deg)
            )
            
            return JSONResponse({
                "success": True,
                "pixels_per_meter": float(result.pixels_per_meter),
                "forward_direction_deg": float(result.forward_direction_deg)
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.error_message
            }, status_code=400)
            
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/tracker/reset")
async def reset_tracker():
    """Reset tracker state."""
    try:
        sim = get_app_instance()
        sim.tracker.reset()
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/measure/distance")
async def measure_distance(data: dict):
    """
    Measure real-world distance between two pixel points.
    
    Use this to validate calibration accuracy with a physical ruler.
    
    Expected data:
    {
        "point1": [x1, y1],  # First point in pixels
        "point2": [x2, y2]   # Second point in pixels
    }
    
    Returns:
    {
        "distance_px": float,      # Distance in pixels
        "distance_m": float,       # Distance in meters (if calibrated)
        "distance_cm": float,      # Distance in centimeters
        "calibrated": bool,        # Whether calibration was used
        "point1_world": [x, y],    # First point in world coords (meters)
        "point2_world": [x, y]     # Second point in world coords (meters)
    }
    """
    try:
        sim = get_app_instance()
        
        p1 = data["point1"]
        p2 = data["point2"]
        
        # Pixel distance
        dx_px = p2[0] - p1[0]
        dy_px = p2[1] - p1[1]
        distance_px = np.sqrt(dx_px**2 + dy_px**2)
        
        # Get best available pixels_per_meter
        pixels_per_meter = sim.get_pixels_per_meter()
        is_auto = sim.auto_calibrator.is_calibrated
        is_manual = sim.calibrator.is_calibrated
        
        # World distance using simple scale (works for overhead camera)
        distance_m = distance_px / pixels_per_meter
        
        # Return measurement using best available calibration
        return JSONResponse({
            "success": True,
            "distance_px": round(distance_px, 1),
            "distance_m": round(distance_m, 4),
            "distance_cm": round(distance_m * 100, 2),
            "calibrated": is_auto or is_manual,
            "auto_calibrated": is_auto,
            "pixels_per_meter": round(pixels_per_meter, 1)
        })
        
    except Exception as e:
        logger.error(f"Distance measurement failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/measure/last-shot")
async def get_last_shot_distance():
    """
    Get the distance traveled by the last detected shot.
    
    Returns start position, end position, and total distance in real-world units.
    Use this to validate that a 30cm putt registers as 30cm.
    """
    try:
        sim = get_app_instance()
        state = sim._current_state
        
        if not state or not state.shot_result:
            return JSONResponse({
                "success": False,
                "error": "No shot recorded yet"
            })
        
        trajectory = state.shot_result.trajectory
        if len(trajectory) < 2:
            return JSONResponse({
                "success": False,
                "error": "Trajectory too short"
            })
        
        # Get best available calibration
        pixels_per_meter = sim.get_pixels_per_meter()
        is_calibrated = sim.auto_calibrator.is_calibrated or sim.calibrator.is_calibrated
        
        # Start and end points in pixels
        start_px = trajectory[0]
        end_px = trajectory[-1]
        
        # Total distance in pixels
        dx_px = end_px[0] - start_px[0]
        dy_px = end_px[1] - start_px[1]
        distance_px = np.sqrt(dx_px**2 + dy_px**2)
        
        # Convert to meters using best available calibration
        distance_m = distance_px / pixels_per_meter
        speed_m_s = state.shot_result.initial_speed_px_s / pixels_per_meter
        
        return JSONResponse({
            "success": True,
            "start_px": [round(start_px[0], 1), round(start_px[1], 1)],
            "end_px": [round(end_px[0], 1), round(end_px[1], 1)],
            "distance_px": round(distance_px, 1),
            "distance_m": round(distance_m, 4),
            "distance_cm": round(distance_m * 100, 2),
            "speed_m_s": round(speed_m_s, 3),
            "direction_deg": round(state.shot_result.initial_direction_deg, 2),
            "duration_ms": round(state.shot_result.duration_ms, 1),
            "calibrated": is_calibrated,
            "auto_calibrated": sim.auto_calibrator.is_calibrated,
            "pixels_per_meter": round(pixels_per_meter, 1)
        })
        
    except Exception as e:
        logger.error(f"Shot distance measurement failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/calibrate/correct-scale")
async def correct_calibration_scale(data: dict):
    """
    Correct the calibration scale based on a known measurement.
    
    Use after measuring with the ruler tool - enter the actual physical
    distance to correct the calibration.
    
    Expected data:
    {
        "measured_cm": float,  # What the system measured
        "actual_cm": float     # What it should be
    }
    """
    try:
        sim = get_app_instance()
        
        measured_cm = data.get("measured_cm")
        actual_cm = data.get("actual_cm")
        
        if measured_cm is None or actual_cm is None:
            return JSONResponse({
                "success": False,
                "error": "Need both measured_cm and actual_cm"
            }, status_code=400)
        
        if measured_cm <= 0 or actual_cm <= 0:
            return JSONResponse({
                "success": False,
                "error": "Values must be positive"
            }, status_code=400)
        
        # Calculate correction factor
        # If measured is 37.2 but actual is 30, scale = 30/37.2 = 0.806
        scale_factor = actual_cm / measured_cm
        
        old_ppm = sim.calibrator.pixels_per_meter
        
        # Scale the homography matrix if it exists
        if sim.calibrator._homography is not None:
            # Scale the homography to correct world coordinates
            # H transforms pixel -> world. If world is too big, multiply H by scale_factor
            H = sim.calibrator._homography.copy()
            # Scale the translation and rotation parts (first 2 rows affect x,y output)
            H[0, :] *= scale_factor  # Scale x output
            H[1, :] *= scale_factor  # Scale y output
            sim.calibrator._homography = H
            
            # Also update pixels_per_meter to match
            new_ppm = old_ppm / scale_factor
            sim.calibrator._pixels_per_meter = new_ppm
            
            # Save to config
            config_mgr = get_config_manager()
            config = config_mgr.config
            config.calibration.homography_matrix = [[float(v) for v in row] for row in H.tolist()]
            config.calibration.pixels_per_meter = float(new_ppm)
            config_mgr.save()
            
            logger.info(f"Calibration scale corrected by factor {scale_factor:.4f}: "
                       f"{old_ppm:.1f} -> {new_ppm:.1f} px/m")
            
            return JSONResponse({
                "success": True,
                "scale_factor": round(scale_factor, 4),
                "old_pixels_per_meter": round(old_ppm, 1),
                "new_pixels_per_meter": round(new_ppm, 1),
                "message": f"Scaled calibration by {scale_factor:.3f}x"
            })
        else:
            # No homography - just update pixels_per_meter
            new_ppm = old_ppm / scale_factor
            sim.calibrator._pixels_per_meter = new_ppm
            
            config_mgr = get_config_manager()
            config = config_mgr.config
            config.calibration.pixels_per_meter = float(new_ppm)
            config_mgr.save()
            
            return JSONResponse({
                "success": True,
                "scale_factor": round(scale_factor, 4),
                "old_pixels_per_meter": round(old_ppm, 1),
                "new_pixels_per_meter": round(new_ppm, 1)
            })
        
    except Exception as e:
        logger.error(f"Calibration correction failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/calibrate/lens-status")
async def get_lens_calibration_status():
    """Get lens distortion calibration status."""
    try:
        sim = get_app_instance()
        
        result = load_lens_calibration()
        
        return JSONResponse({
            "calibrated": sim._lens_calibrated,
            "file_exists": LENS_PARAMS_FILE.exists(),
            "reprojection_error": result.reprojection_error if result else None,
            "num_images_used": result.num_images_used if result else None,
            "calibrated_at": result.calibrated_at if result else None,
            "instructions": "Run 'python -m backend.lens_calibration --live' to calibrate" if not sim._lens_calibrated else None
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/calibrate/verify")
async def verify_calibration(data: dict):
    """
    Verify calibration accuracy by comparing measured vs actual distance.
    
    Place the ball at a known distance from the start position, then call this
    endpoint with the actual physical distance to compare.
    
    Expected data:
    {
        "actual_distance_cm": 50.0  # The real physical distance in cm
    }
    
    Returns comparison between system measurement and actual distance.
    """
    try:
        sim = get_app_instance()
        state = sim._current_state
        
        actual_cm = data.get("actual_distance_cm")
        if actual_cm is None:
            return JSONResponse({
                "success": False,
                "error": "actual_distance_cm is required"
            }, status_code=400)
        
        if not state or not state.shot_result:
            return JSONResponse({
                "success": False,
                "error": "No shot recorded. Make a putt first, then verify."
            })
        
        # Get measured distance
        pixels_per_meter = sim.get_pixels_per_meter()
        total_distance_px = state.shot_result.total_distance_px
        measured_cm = (total_distance_px / pixels_per_meter) * 100
        
        # Calculate error
        error_cm = measured_cm - actual_cm
        error_percent = (error_cm / actual_cm) * 100 if actual_cm > 0 else 0
        
        # Determine accuracy rating
        if abs(error_percent) < 3:
            rating = "excellent"
        elif abs(error_percent) < 5:
            rating = "good"
        elif abs(error_percent) < 10:
            rating = "acceptable"
        else:
            rating = "needs_calibration"
        
        return JSONResponse({
            "success": True,
            "actual_cm": round(actual_cm, 1),
            "measured_cm": round(measured_cm, 1),
            "error_cm": round(error_cm, 2),
            "error_percent": round(error_percent, 2),
            "accuracy_rating": rating,
            "pixels_per_meter": round(pixels_per_meter, 1),
            "lens_calibrated": sim._lens_calibrated,
            "suggestion": (
                "Consider running lens calibration" if not sim._lens_calibrated and abs(error_percent) > 5
                else "Calibration looks good" if abs(error_percent) < 5
                else f"Try adjusting distance_scale_factor to {round(actual_cm / measured_cm, 3)}" if measured_cm > 0
                else "Unable to suggest adjustment"
            )
        })
        
    except Exception as e:
        logger.error(f"Calibration verification failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/diagnostics")
async def get_diagnostics():
    """
    Get comprehensive diagnostic information for debugging accuracy issues.
    
    Returns detailed state about calibration, tracking, and recent shots.
    """
    try:
        sim = get_app_instance()
        state = sim._current_state
        
        # Build diagnostic report
        diagnostics = {
            "calibration": {
                "lens_calibrated": sim._lens_calibrated,
                "auto_calibrated": sim.auto_calibrator.is_calibrated,
                "manual_calibrated": sim.calibrator.is_calibrated,
                "current_pixels_per_meter": round(sim.get_pixels_per_meter(), 1),
                "shot_locked_ppm": round(sim._shot_locked_ppm, 1) if sim._shot_locked_ppm else None,
                "auto_calibrator_confidence": round(sim.auto_calibrator.confidence, 3) if sim.auto_calibrator.is_calibrated else None
            },
            "tracker": {
                "state": state.state.value if state else "unknown",
                "lane": state.lane.value if state else "unknown",
                "forward_direction_deg": round(sim.tracker._forward_direction_deg, 1),
                "valid_motion_angle_deg": round(sim.tracker._valid_motion_angle_deg, 1),
                "deceleration_m_s2": round(sim.tracker.DECELERATION_M_S2, 3),
                "deceleration_px_s2": round(sim.tracker.get_deceleration_px_s2(), 1)
            },
            "last_shot": None,
            "system": {
                "camera_resolution": list(sim.camera.resolution) if sim.camera else None,
                "cap_fps": round(sim._cap_fps.tick(), 1),
                "proc_fps": round(sim._proc_fps.tick(), 1)
            }
        }
        
        # Add last shot details if available
        if state and state.shot_result:
            result = state.shot_result
            ppm = sim.get_pixels_per_meter()
            
            diagnostics["last_shot"] = {
                "initial_speed_px_s": round(result.initial_speed_px_s, 1),
                "initial_speed_m_s": round(result.initial_speed_px_s / ppm, 3),
                "direction_deg": round(result.initial_direction_deg, 2),
                "physical_distance_px": round(result.total_distance_px - result.virtual_distance_px, 1),
                "virtual_distance_px": round(result.virtual_distance_px, 1),
                "total_distance_px": round(result.total_distance_px, 1),
                "total_distance_m": round(result.total_distance_px / ppm, 4),
                "total_distance_cm": round((result.total_distance_px / ppm) * 100, 1),
                "exited_frame": result.exited_frame,
                "duration_ms": round(result.duration_ms, 1),
                "trajectory_points": len(result.trajectory)
            }
        
        return JSONResponse(diagnostics)
        
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/calibrate/detect-aruco")
async def detect_aruco():
    """Detect ArUco markers in current frame."""
    try:
        from .calibration import detect_aruco_markers
        
        sim = get_app_instance()
        if sim._current_frame is None:
            return JSONResponse({"success": False, "error": "No frame available"})
        
        corners, ids = detect_aruco_markers(sim._current_frame)
        
        markers = []
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0].tolist()
            center = [
                sum(c[0] for c in marker_corners) / 4,
                sum(c[1] for c in marker_corners) / 4
            ]
            markers.append({
                "id": marker_id,
                "corners": marker_corners,
                "center": center
            })
        
        return JSONResponse({
            "success": True,
            "markers": markers,
            "count": len(markers)
        })
        
    except Exception as e:
        logger.error(f"ArUco detection failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/calibrate/aruco")
async def calibrate_aruco(data: dict):
    """
    Calibrate using detected ArUco markers.
    
    Expected data:
    {
        "marker_positions": {
            "0": [0, 0],      # marker_id: [x_meters, y_meters]
            "1": [0.5, 0],
            "2": [0.5, 1.0],
            "3": [0, 1.0]
        },
        "marker_size_m": 0.05
    }
    """
    try:
        sim = get_app_instance()
        if sim._current_frame is None:
            return JSONResponse({"success": False, "error": "No frame available"})
        
        marker_world_positions = {
            int(k): tuple(v) for k, v in data["marker_positions"].items()
        }
        marker_size_m = data.get("marker_size_m", 0.05)
        
        result = sim.calibrator.calibrate_from_aruco(
            sim._current_frame,
            marker_world_positions,
            marker_size_m
        )
        
        if result.success:
            config_mgr = get_config_manager()
            config_mgr.update_calibration(
                homography_matrix=result.homography_matrix.tolist(),
                pixels_per_meter=result.pixels_per_meter,
                origin_px=result.origin_px,
                forward_direction_deg=result.forward_direction_deg
            )
            
            return JSONResponse({
                "success": True,
                "pixels_per_meter": result.pixels_per_meter,
                "forward_direction_deg": result.forward_direction_deg
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result.error_message
            }, status_code=400)
            
    except Exception as e:
        logger.error(f"ArUco calibration failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time state updates."""
    await websocket.accept()
    
    try:
        sim = get_app_instance()
        await sim.add_client(websocket)
        
        # Keep connection alive
        while True:
            try:
                # Wait for any message (keepalive)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                # Handle commands
                try:
                    cmd = json.loads(data)
                    if cmd.get("type") == "reset":
                        sim.tracker.reset()
                    elif cmd.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
                    
            except asyncio.TimeoutError:
                # Send ping to check connection
                await websocket.send_text(json.dumps({"type": "ping"}))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await sim.remove_client(websocket)
        except:
            pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="StrikeLab Putting Sim")
    parser.add_argument("--arducam", action="store_true", help="Use Arducam OV9281")
    parser.add_argument("--webcam", action="store_true", help="Use standard webcam")
    parser.add_argument("--replay", type=str, help="Replay from video file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine camera mode
    if args.replay:
        camera_mode = CameraMode.REPLAY
        replay_path = args.replay
    elif args.webcam:
        camera_mode = CameraMode.WEBCAM
        replay_path = None
    else:
        camera_mode = CameraMode.ARDUCAM
        replay_path = None
    
    # Create app instance
    global app_instance
    app_instance = PuttingSimApp(
        camera_mode=camera_mode,
        replay_path=replay_path
    )
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Camera mode: {camera_mode.value}")
    if replay_path:
        logger.info(f"Replay file: {replay_path}")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info" if not args.debug else "debug"
    )


if __name__ == "__main__":
    main()
