"""
StrikeLab Putting Sim - Main entry point.
FastAPI server with WebSocket streaming and camera capture loop.
"""

import argparse
import asyncio
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
from .tracker import BallTracker, TrackerState, ShotState
from .calibration import Calibrator
from .predictor import BallPredictor
from .config import get_config_manager, get_config

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
        
        # Components
        self.camera: Optional[Camera] = None
        self.detector = BallDetector()
        self.tracker = BallTracker()
        self.calibrator = Calibrator()
        self.predictor = BallPredictor()
        
        # State
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._current_state: Optional[TrackerState] = None
        self._current_frame: Optional[np.ndarray] = None
        self._frame_id = 0
        
        # FPS tracking
        self._cap_fps = FPSTracker()
        self._proc_fps = FPSTracker()
        self._disp_fps = FPSTracker()
        self._last_proc_time = 0.0
        
        # WebSocket clients
        self._ws_clients: Set[WebSocket] = set()
        self._ws_lock = asyncio.Lock()
        
        # Load config
        config = get_config()
        if config.calibration.is_valid():
            self.calibrator.load_from_config({
                "homography_matrix": config.calibration.homography_matrix,
                "pixels_per_meter": config.calibration.pixels_per_meter,
                "origin_px": config.calibration.origin_px,
                "forward_direction_deg": config.calibration.forward_direction_deg
            })
    
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
            self._current_frame = frame_data.frame
    
    def _process_frame(self, frame_data: FrameData):
        """Process a single frame."""
        # Detect ball
        detection = self.detector.detect(frame_data.frame)
        
        # Update tracker (simplified - no frame passing to avoid background model overhead)
        self._current_state = self.tracker.update(
            detection,
            frame_data.timestamp_ns,
            frame_data.frame_id
        )
    
    def get_state_message(self) -> dict:
        """Build state message for WebSocket."""
        state = self._current_state
        
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
            ball_visible = (
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
        
        # Prediction data (when ball exits frame or after shot)
        prediction_data = None
        if state and state.velocity and not ball_visible:
            # Ball has exited - generate prediction
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
        
        # Shot result
        shot_data = None
        if state and state.shot_result:
            result = state.shot_result
            
            # Convert to world coordinates if calibrated
            speed_m_s = result.initial_speed_px_s / self.calibrator.pixels_per_meter
            direction_deg = self.calibrator.direction_relative_to_forward(result.initial_direction_deg)
            
            shot_data = {
                "speed_m_s": round(speed_m_s, 3),
                "speed_px_s": round(result.initial_speed_px_s, 1),
                "direction_deg": round(direction_deg, 2),
                "frames_to_tracking": result.frames_to_tracking,
                "frames_to_speed": result.frames_to_speed,
                "duration_ms": round(result.duration_ms, 1),
                "trajectory": result.trajectory[-50:]  # Last 50 points
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
            "shot": shot_data,
            "metrics": metrics,
            "calibrated": self.calibrator.is_calibrated,
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
    """MJPEG video stream for viewing camera output at ~30fps."""
    
    def generate_frames():
        import cv2
        sim = get_app_instance()
        last_frame_time = 0
        
        while sim._running:
            # Limit to ~30fps for browser display
            now = time.time()
            if now - last_frame_time < 1/30:
                time.sleep(0.005)
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
                
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
