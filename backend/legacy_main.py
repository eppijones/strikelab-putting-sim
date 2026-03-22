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
from typing import Optional, Set, Any, Dict
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .camera import Camera, CameraMode, FrameData
from .detector import BallDetector, Detection
from .tracker import BallTracker, TrackerState, ShotState, VirtualBallState
from .calibration import Calibrator, AutoCalibrator
from .predictor import BallPredictor
from .config import get_config_manager, get_config
from .lens_calibration import load_lens_calibration, get_undistort_maps, LENS_PARAMS_FILE
from .game_logic import GameLogic, get_game_logic, ShotResult as GameShotResult
from .session import SessionManager, get_session_manager
from .drills import DrillManager, get_drill_manager, DrillType

# Multi-camera system
from .cameras.camera_manager import CameraManager, CameraType
from .tracking.ball_tracker_3d import BallTracker3D
from .tracking.club_tracker import ClubTracker
from .tracking.launch_detector import LaunchDetector
from .tracking.sensor_fusion import SensorFusion, SensorFusionPolicy
from .analysis.putting_analyzer import PuttingAnalyzer
from .analysis.chipping_analyzer import ChippingAnalyzer
from .cameras.arducam_source import find_arducam_device  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ARDUCAM_TARGET_FPS = 75.0
ARDUCAM_MIN_HEALTHY_FPS_RATIO = 0.70
ARDUCAM_MAX_CONSECUTIVE_READ_FAILURES = 360  # ~3.0s at 120fps
ARDUCAM_LOW_FPS_GRACE_S = 5.0
ARDUCAM_RESTART_COOLDOWN_S = 5.0
ARDUCAM_RESTART_ON_LOW_FPS = False
ARDUCAM_LOW_FPS_WARN_COOLDOWN_S = 60.0
STALE_MANUAL_CALIB_WARN_COOLDOWN_S = 30.0
MAX_REASONABLE_LENS_K1 = 1.5
MAX_REASONABLE_LENS_K2 = 2.0
MAX_REASONABLE_LENS_K3 = 2.0
MAX_REASONABLE_LENS_P1P2 = 0.25
ARDUCAM_TELEMETRY_LOG_INTERVAL_S = 10.0


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
            stopped_confirm_time_ms=getattr(config.tracker, 'stopped_confirm_time_ms', 100),
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
        self._depth_reconnect_thread: Optional[threading.Thread] = None
        self._depth_reconnect_interval_s: float = 10.0
        self._current_state: Optional[TrackerState] = None
        self._current_frame: Optional[np.ndarray] = None
        self._frame_id = 0
        
        # Locked calibration for completed shots (prevents distance changing after STOPPED)
        self._shot_locked_ppm: Optional[float] = None
        self._last_shot_state: Optional[ShotState] = None
        self._active_calibration_ppm: float = float(config.calibration.pixels_per_meter or 1150.0)
        self._active_calibration_confidence: float = 0.0
        self._active_calibration_source: str = "config"
        
        # Game logic and session tracking
        self.game_logic = get_game_logic()
        self.session_manager = get_session_manager()
        self.drill_manager = get_drill_manager()
        self._previous_game_state: Optional[ShotState] = None
        self._shot_analyzed = False  # Flag to prevent duplicate analysis
        
        # FPS tracking
        self._cap_fps = FPSTracker()
        self._proc_fps = FPSTracker()
        self._disp_fps = FPSTracker()
        self._last_proc_time = 0.0
        self._last_proc_frame_age_ms = 0.0
        self._proc_frame_age_ema_ms = 0.0

        # Arducam low-latency handoff: capture thread writes latest frame, process thread consumes newest.
        self._process_thread: Optional[threading.Thread] = None
        self._arducam_latest_lock = threading.Lock()
        self._arducam_latest_frame: Optional[FrameData] = None
        self._arducam_frame_ready = threading.Event()
        self._arducam_capture_count: int = 0
        self._arducam_processed_count: int = 0
        self._arducam_dropped_count: int = 0
        self._arducam_backlog_peak: int = 0
        self._arducam_last_capture_seq: int = 0
        self._arducam_last_telemetry_log_s: float = 0.0
        
        # WebSocket clients
        self._ws_clients: Set[WebSocket] = set()
        self._ws_lock = asyncio.Lock()
        
        # --- Multi-camera system ---
        mc_config = config.multi_camera
        self.camera_manager = CameraManager(
            enable_arducam=True,
            enable_zed=mc_config.zed.enabled,
            enable_realsense=mc_config.realsense.enabled,
            arducam_device_id=config.camera.device_id,
            arducam_replay_path=replay_path if camera_mode == CameraMode.REPLAY else None,
            zed_serial=mc_config.zed.serial_number,
            realsense_serial=mc_config.realsense.serial_number or None,
            zed_settings={
                "resolution": mc_config.zed.resolution,
                "fps": mc_config.zed.fps,
                "depth_mode": mc_config.zed.depth_mode,
                "min_depth_m": mc_config.zed.min_depth_m,
                "max_depth_m": mc_config.zed.max_depth_m,
                "confidence_threshold": mc_config.zed.confidence_threshold,
                "auto_exposure_gain": mc_config.zed.auto_exposure_gain,
                "auto_white_balance": mc_config.zed.auto_white_balance,
                "exposure": mc_config.zed.exposure,
                "gain": mc_config.zed.gain,
                "whitebalance_temperature": mc_config.zed.whitebalance_temperature,
                "brightness": mc_config.zed.brightness,
                "contrast": mc_config.zed.contrast,
                "saturation": mc_config.zed.saturation,
                "sharpness": mc_config.zed.sharpness,
                "gamma": mc_config.zed.gamma,
            },
            realsense_settings={
                "depth_width": mc_config.realsense.depth_width,
                "depth_height": mc_config.realsense.depth_height,
                "color_width": mc_config.realsense.color_width,
                "color_height": mc_config.realsense.color_height,
                "fps": mc_config.realsense.fps,
                "depth_visual_preset": mc_config.realsense.depth_visual_preset,
                "depth_auto_exposure": mc_config.realsense.depth_auto_exposure,
                "emitter_enabled": mc_config.realsense.emitter_enabled,
                "laser_power": mc_config.realsense.laser_power,
                "enable_depth_post_processing": mc_config.realsense.enable_depth_post_processing,
                "color_auto_exposure": mc_config.realsense.color_auto_exposure,
                "color_exposure": mc_config.realsense.color_exposure,
                "color_gain": mc_config.realsense.color_gain,
                "color_auto_white_balance": mc_config.realsense.color_auto_white_balance,
                "color_white_balance": mc_config.realsense.color_white_balance,
                "color_sharpness": mc_config.realsense.color_sharpness,
                "color_contrast": mc_config.realsense.color_contrast,
                "color_saturation": mc_config.realsense.color_saturation,
                "color_brightness": mc_config.realsense.color_brightness,
            },
        )
        
        # Depth camera trackers
        self.ball_tracker_3d = BallTracker3D(
            surface_height_m=mc_config.zed.surface_height_m,
        )
        self.club_tracker = ClubTracker(
            surface_depth_m=mc_config.zed.surface_height_m,
        )
        self.launch_detector = LaunchDetector()
        
        # Sensor fusion
        self.sensor_fusion = SensorFusion(pixels_per_meter=config.calibration.pixels_per_meter)
        self._sync_sensor_fusion_policy()
        self.putting_analyzer = PuttingAnalyzer()
        self.chipping_analyzer = ChippingAnalyzer()
        
        # Multi-camera state
        self._zed_frame: Optional[np.ndarray] = None
        self._rs_frame: Optional[np.ndarray] = None
        self._latest_fused_report = None
        self._latest_shot_report = None
        self._last_arducam_frame_monotonic_ns: Optional[int] = None
        self._last_arducam_detection_monotonic_ns: Optional[int] = None
        self._last_zed_detection_monotonic_ns: Optional[int] = None
        self._last_realsense_detection_monotonic_ns: Optional[int] = None
        self._ws_broadcast_seq: int = 0
        self._arducam_consecutive_read_failures: int = 0
        self._arducam_low_fps_since: Optional[float] = None
        self._arducam_last_restart_monotonic_s: float = 0.0
        self._arducam_last_low_fps_warn_monotonic_s: float = 0.0
        self._arducam_target_fps: float = ARDUCAM_TARGET_FPS
        self._last_stale_manual_calib_warn_monotonic_s: float = 0.0

        # Startup readiness tracking
        self._startup_phase: str = "starting"
        self._startup_arducam_profile_info: Optional[str] = None
        self._startup_sustained_fps: float = 0.0
        self._startup_fail_reason: Optional[str] = None
        
        # Wire up fast putt speed resolution from depth cameras
        def _resolve_fast_putt_speed(timestamp_ns):
            resolved = self.sensor_fusion.resolve_fast_putt(timestamp_ns)
            if resolved.confidence > 0.5 and resolved.source != "estimated":
                return (resolved.vx_px_s, resolved.vy_px_s)
            return None
        
        self.tracker.set_fast_putt_speed_callback(_resolve_fast_putt_speed)
        
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
            camera_matrix = np.array(lens_result.camera_matrix)
            dist_coeffs = np.array(lens_result.dist_coeffs)
            if self._is_lens_calibration_plausible(dist_coeffs):
                self._camera_matrix = camera_matrix
                self._dist_coeffs = dist_coeffs
                self._lens_calibrated = True
                logger.info(f"Lens calibration loaded (error: {lens_result.reprojection_error:.3f}px)")
            else:
                self._lens_calibrated = False
                logger.warning(
                    "Lens calibration coefficients look unstable; disabling undistortion for this run. "
                    "Running in fallback mode (no undistortion, conservative detector assumptions). "
                    "Re-run lens calibration to restore correction."
                )
        else:
            logger.info("No lens calibration - run 'python -m backend.lens_calibration --live' to calibrate")
    
    def reset_all(self):
        """Fully reset tracker, shot report, game analysis, and fusion caches."""
        self.tracker.reset()
        self._latest_shot_report = None
        self._latest_fused_report = None
        self._shot_locked_ppm = None
        self._shot_analyzed = False
        self.game_logic._last_analysis = None
        self.ball_tracker_3d.reset()
        self.club_tracker.reset()
        self.launch_detector.reset()
        self.sensor_fusion.reset()
        logger.info("Full reset: tracker, shot report, game analysis, and fusion caches cleared")

    def _sync_sensor_fusion_policy(self) -> None:
        """Apply multi_camera config to SensorFusionPolicy."""
        mc = get_config().multi_camera
        self.sensor_fusion.set_policy(
            SensorFusionPolicy(
                enable_speed_fusion=getattr(mc, "enable_speed_fusion", True),
                weight_arducam=float(getattr(mc, "fusion_weight_arducam", 0.5)),
                weight_zed=float(getattr(mc, "fusion_weight_zed", 0.35)),
                weight_realsense=float(getattr(mc, "fusion_weight_realsense", 0.15)),
                speed_inconsistency_threshold_m_s=float(
                    getattr(mc, "speed_inconsistency_threshold_m_s", 0.45)
                ),
                sync_tolerance_ms=int(getattr(mc, "sync_tolerance_ms", 20)),
                enable_direction_fusion=bool(getattr(mc, "enable_direction_fusion", False)),
                allow_realsense_speed_fusion=bool(getattr(mc, "allow_realsense_speed_fusion", False)),
                sensor_direction_alignment_valid=bool(getattr(mc, "sensor_direction_alignment_valid", False)),
            )
        )

    @staticmethod
    def _is_expected_arducam_resolution(width: int, height: int) -> bool:
        return abs(width - 1280) <= 64 and abs(height - 800) <= 64

    @staticmethod
    def _is_lens_calibration_plausible(dist_coeffs: np.ndarray) -> bool:
        """
        Guardrail: reject clearly overfit/unstable distortion coefficients that can
        severely warp the live image. We can still track without lens correction.
        """
        coeffs = np.array(dist_coeffs, dtype=np.float64).reshape(-1)
        if coeffs.size < 4:
            return False

        k1 = abs(float(coeffs[0]))
        k2 = abs(float(coeffs[1]))
        p1 = abs(float(coeffs[2]))
        p2 = abs(float(coeffs[3]))
        k3 = abs(float(coeffs[4])) if coeffs.size >= 5 else 0.0

        return (
            k1 <= MAX_REASONABLE_LENS_K1
            and k2 <= MAX_REASONABLE_LENS_K2
            and k3 <= MAX_REASONABLE_LENS_K3
            and p1 <= MAX_REASONABLE_LENS_P1P2
            and p2 <= MAX_REASONABLE_LENS_P1P2
        )

    def _start_primary_arducam_with_fallback(self, preferred_device_id: int) -> bool:
        """
        Start the primary Arducam with device-index fallback.

        Tries the preferred device first (no separate probe scan).  Camera.start()
        internally validates resolution & stereo detection, and also tries MSMF as
        a fallback backend if DSHOW fps is low.  If the preferred device fails, we
        iterate 0-7.
        """
        candidates = [preferred_device_id] + [
            i for i in range(8) if i != preferred_device_id
        ]

        for device_id in candidates:
            cam = Camera(
                mode=self.camera_mode,
                replay_path=self.replay_path,
                device_id=device_id,
            )
            if not cam.start():
                continue
            w, h = cam.resolution
            if self._is_expected_arducam_resolution(w, h):
                self.camera = cam
                logger.info(
                    f"Primary Arducam started on device {device_id} at {w}x{h}"
                )
                return True
            logger.warning(
                f"Device {device_id} opened at {w}x{h} (not Arducam target); "
                "trying next device"
            )
            cam.stop()

        logger.error(
            "No 1280x800 Arducam stream found on any device 0-7. "
            "Check USB camera ordering/cables."
        )
        return False
    
    def _restart_primary_arducam(self, reason: str) -> bool:
        """
        Attempt a live Arducam restart while keeping the app running.
        This improves robustness when UVC streams stall under USB load.
        """
        if self.camera_mode != CameraMode.ARDUCAM:
            return False
        now = time.monotonic()
        if now - self._arducam_last_restart_monotonic_s < ARDUCAM_RESTART_COOLDOWN_S:
            return False
        self._arducam_last_restart_monotonic_s = now

        logger.warning(f"Arducam recovery triggered: {reason}")
        config = get_config()
        try:
            if self.camera:
                self.camera.stop()
        except Exception:
            pass
        self.camera = None
        self._last_arducam_frame_monotonic_ns = None
        self._arducam_consecutive_read_failures = 0
        self._arducam_low_fps_since = None
        self._arducam_last_low_fps_warn_monotonic_s = 0.0
        self._cap_fps = FPSTracker()
        self._proc_fps = FPSTracker()
        self._arducam_capture_count = 0
        self._arducam_processed_count = 0
        self._arducam_dropped_count = 0
        self._arducam_backlog_peak = 0
        with self._arducam_latest_lock:
            self._arducam_latest_frame = None

        if not self._start_primary_arducam_with_fallback(config.camera.device_id):
            logger.error("Arducam recovery failed; stream remains offline")
            return False
        logger.info("Arducam recovery succeeded")
        return True

    def start(self):
        """Start camera capture and processing."""
        if self._running:
            return

        self._startup_phase = "cameras_init"
        logger.info(f"Starting PuttingSim with mode={self.camera_mode.value}")
        config = get_config()
        if self.camera_mode == CameraMode.ARDUCAM:
            if not self._start_primary_arducam_with_fallback(config.camera.device_id):
                logger.error("Failed to start primary Arducam camera")
                self._startup_phase = "error"
                self._startup_fail_reason = "arducam_start_failed"
                return
        else:
            self.camera = Camera(
                mode=self.camera_mode,
                replay_path=self.replay_path,
                device_id=config.camera.device_id,
            )
            if not self.camera.start():
                logger.error("Failed to start camera")
                self._startup_phase = "error"
                self._startup_fail_reason = "camera_start_failed"
                return

        # Store Arducam profile info for telemetry
        if self.camera:
            w, h = self.camera.resolution
            sustained = getattr(self.camera, '_sustained_startup_fps', 0.0)
            self._startup_arducam_profile_info = f"{w}x{h}"
            self._startup_sustained_fps = sustained

        self._startup_phase = "camera_ready"
        logger.info("Camera initialisation complete — starting processing threads")

        self._running = True
        self._cap_fps = FPSTracker()
        self._proc_fps = FPSTracker()
        self._arducam_capture_count = 0
        self._arducam_processed_count = 0
        self._arducam_dropped_count = 0
        self._arducam_backlog_peak = 0
        self._last_proc_frame_age_ms = 0.0
        self._proc_frame_age_ema_ms = 0.0
        with self._arducam_latest_lock:
            self._arducam_latest_frame = None
        self._arducam_frame_ready.clear()

        # Start primary Arducam capture + processing threads.
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="cam-arducam-capture")
        self._capture_thread.start()
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True, name="cam-arducam-process")
        self._process_thread.start()
        
        # Start depth cameras via CameraManager (non-blocking, graceful degradation)
        self._start_depth_cameras()
        self._start_depth_reconnect_worker()

        self._startup_phase = "running"
        logger.info("PuttingSim started — startup_phase=running")
    
    def _start_depth_cameras(self):
        """Start ZED and RealSense cameras in background. Failures are non-fatal."""
        config = get_config()
        mc = config.multi_camera
        
        # Register callbacks for depth camera frames
        def on_zed_frame(frame):
            self._process_zed_frame(frame)
        
        def on_rs_frame(frame):
            self._process_realsense_frame(frame)
        
        if mc.zed.enabled:
            self.camera_manager.on_frame(CameraType.ZED, on_zed_frame)
        if mc.realsense.enabled:
            self.camera_manager.on_frame(CameraType.REALSENSE, on_rs_frame)
        
        # Start depth cameras through CameraManager so status + sync buffers stay accurate.
        start_types = []
        if mc.zed.enabled:
            start_types.append(CameraType.ZED)
        if mc.realsense.enabled:
            start_types.append(CameraType.REALSENSE)
        results = self.camera_manager.start_selected(start_types)
        if mc.zed.enabled and not results.get(CameraType.ZED, False):
            logger.warning("ZED 2i failed to start - continuing without it")
        if mc.realsense.enabled and not results.get(CameraType.REALSENSE, False):
            logger.warning("RealSense D455 failed to start - continuing without it")
        
        active = [ct.value for ct, ok in results.items() if ok]
        if active:
            logger.info(f"Depth cameras active: {', '.join(active)}")
        else:
            logger.info("No depth cameras active - running with Arducam only")

    def _start_depth_reconnect_worker(self) -> None:
        if self._depth_reconnect_thread and self._depth_reconnect_thread.is_alive():
            return
        self._depth_reconnect_thread = threading.Thread(
            target=self._depth_reconnect_loop,
            daemon=True,
            name="cam-depth-reconnect",
        )
        self._depth_reconnect_thread.start()

    def _depth_reconnect_loop(self) -> None:
        """
        Keep trying to bring up enabled depth cameras that failed at startup.
        This makes camera bring-up resilient to USB timing/driver init order.
        """
        while self._running:
            time.sleep(self._depth_reconnect_interval_s)
            if not self._running:
                break
            try:
                mc = get_config().multi_camera
                to_start = []
                if mc.zed.enabled:
                    zed_src = self.camera_manager.get_source(CameraType.ZED)
                    if zed_src and not zed_src.is_running:
                        to_start.append(CameraType.ZED)
                if mc.realsense.enabled:
                    rs_src = self.camera_manager.get_source(CameraType.REALSENSE)
                    if rs_src and not rs_src.is_running:
                        to_start.append(CameraType.REALSENSE)
                if not to_start:
                    continue
                results = self.camera_manager.start_selected(to_start)
                for ct, ok in results.items():
                    if ok:
                        logger.info(f"{ct.value} recovered and is now active")
            except Exception as e:
                logger.debug(f"Depth reconnect loop error: {e}")
    
    def _depth_capture_loop(self, camera_type: CameraType):
        """Capture loop for a depth camera."""
        source = self.camera_manager.get_source(camera_type)
        if not source:
            return
        while self._running and source.is_running:
            frame = source.read_frame()
            if frame is None:
                continue
            source.mark_frame_received()
            for cb in self.camera_manager._callbacks.get(camera_type, []):
                try:
                    cb(frame)
                except Exception as e:
                    logger.error(f"Depth camera callback error ({camera_type.value}): {e}")
    
    def stop(self):
        """Stop capture and cleanup."""
        self._running = False
        self._arducam_frame_ready.set()
        
        if self.camera:
            self.camera.stop()
        
        # Stop depth cameras + join their capture threads.
        self.camera_manager.stop()
        if self._depth_reconnect_thread:
            self._depth_reconnect_thread.join(timeout=2.0)
            self._depth_reconnect_thread = None
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
        
        logger.info("PuttingSim stopped")
    
    def _capture_loop(self):
        """Arducam capture loop: always keep freshest frame."""
        last_watchdog_check = time.monotonic()
        while self._running:
            # Watchdog: restart process thread if it died
            now_mono = time.monotonic()
            if now_mono - last_watchdog_check > 2.0:
                last_watchdog_check = now_mono
                if self._process_thread and not self._process_thread.is_alive():
                    logger.warning("Process thread died — restarting it")
                    self._process_thread = threading.Thread(
                        target=self._process_loop, daemon=True, name="cam-arducam-process"
                    )
                    self._process_thread.start()

            if self.camera is None:
                time.sleep(0.01)
                continue
            frame_data = self.camera.read_frame()
            if frame_data is not None:
                self._last_arducam_frame_monotonic_ns = time.monotonic_ns()
                self._arducam_consecutive_read_failures = 0
            
            if frame_data is None:
                if self.camera_mode == CameraMode.REPLAY:
                    logger.info("Replay finished")
                    self._running = False
                elif self.camera_mode == CameraMode.ARDUCAM:
                    self._arducam_consecutive_read_failures += 1
                    if self._arducam_consecutive_read_failures >= ARDUCAM_MAX_CONSECUTIVE_READ_FAILURES:
                        self._restart_primary_arducam(
                            reason=f"no frames for {self._arducam_consecutive_read_failures} reads"
                        )
                continue

            self._arducam_capture_count += 1

            # Track capture FPS and health.
            cap_fps = self._cap_fps.tick()
            if self.camera_mode == CameraMode.ARDUCAM:
                min_healthy_fps = max(1.0, self._arducam_target_fps * ARDUCAM_MIN_HEALTHY_FPS_RATIO)
                if cap_fps > 0 and cap_fps < min_healthy_fps:
                    if self._arducam_low_fps_since is None:
                        self._arducam_low_fps_since = time.monotonic()
                    elif (time.monotonic() - self._arducam_low_fps_since) > ARDUCAM_LOW_FPS_GRACE_S:
                        if ARDUCAM_RESTART_ON_LOW_FPS:
                            self._restart_primary_arducam(
                                reason=f"capture fps degraded to {cap_fps:.1f} (< {min_healthy_fps:.1f})"
                            )
                        else:
                            now = time.monotonic()
                            if (now - self._arducam_last_low_fps_warn_monotonic_s) > ARDUCAM_LOW_FPS_WARN_COOLDOWN_S:
                                self._arducam_last_low_fps_warn_monotonic_s = now
                                logger.warning(
                                    f"Arducam capture FPS is below target ({cap_fps:.1f} < {min_healthy_fps:.1f}) "
                                    "but stream is stable; skipping auto-restart"
                                )
                else:
                    self._arducam_low_fps_since = None

            with self._arducam_latest_lock:
                if self._arducam_latest_frame is not None:
                    self._arducam_dropped_count += 1
                self._arducam_latest_frame = frame_data
                backlog = max(0, self._arducam_capture_count - self._arducam_processed_count)
                self._arducam_backlog_peak = max(self._arducam_backlog_peak, backlog)
            self._arducam_frame_ready.set()

    def _process_loop(self):
        """Arducam processing loop: consume newest available frame only."""
        consecutive_errors = 0
        while self._running:
            try:
                if not self._arducam_frame_ready.wait(timeout=0.050):
                    continue
                frame_data: Optional[FrameData] = None
                with self._arducam_latest_lock:
                    frame_data = self._arducam_latest_frame
                    self._arducam_latest_frame = None
                    self._arducam_frame_ready.clear()
                if frame_data is None:
                    continue

                proc_start = time.time()
                self._process_frame(frame_data)
                self._last_proc_time = (time.time() - proc_start) * 1000
                self._proc_fps.tick()
                self._arducam_processed_count += 1
                self._frame_id = frame_data.frame_id
                self._last_proc_frame_age_ms = (
                    time.monotonic_ns() - frame_data.acquisition_monotonic_ns
                ) / 1e6
                if self._proc_frame_age_ema_ms <= 0.0:
                    self._proc_frame_age_ema_ms = self._last_proc_frame_age_ms
                else:
                    self._proc_frame_age_ema_ms = (
                        0.9 * self._proc_frame_age_ema_ms + 0.1 * self._last_proc_frame_age_ms
                    )
                self._log_arducam_telemetry()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    logger.error(f"Process loop error (#{consecutive_errors}): {e}", exc_info=True)
                elif consecutive_errors == 4:
                    logger.error("Suppressing further process loop error details")
                if consecutive_errors > 100:
                    time.sleep(0.01)

    def _log_arducam_telemetry(self):
        now = time.monotonic()
        if (now - self._arducam_last_telemetry_log_s) < ARDUCAM_TELEMETRY_LOG_INTERVAL_S:
            return
        self._arducam_last_telemetry_log_s = now
        backlog = max(0, self._arducam_capture_count - self._arducam_processed_count)
        logger.info(
            "Arducam telemetry: capture_fps=%.1f, tracker_fps=%.1f, dropped=%d, backlog=%d, "
            "frame_age_ms(avg)=%.1f, frame_age_ms(last)=%.1f",
            self._fps_snapshot(self._cap_fps),
            self._fps_snapshot(self._proc_fps),
            self._arducam_dropped_count,
            backlog,
            self._proc_frame_age_ema_ms,
            self._last_proc_frame_age_ms,
        )

    @staticmethod
    def _fps_snapshot(tracker: FPSTracker) -> float:
        if len(tracker.timestamps) < 2:
            return 0.0
        dt = tracker.timestamps[-1] - tracker.timestamps[0]
        return (len(tracker.timestamps) - 1) / dt if dt > 0 else 0.0
    
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
        
        # Store undistorted frame for video streaming (avoids double remap)
        self._current_frame = frame
        
        # Detect ball on undistorted frame
        detection = self.detector.detect(frame)
        if detection is not None:
            self._last_arducam_detection_monotonic_ns = time.monotonic_ns()
        
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
            current_ppm = self.get_pixels_per_meter()
            if self._active_calibration_confidence < 0.5:
                logger.warning(
                    "Calibration confidence too low at shot start (source=%s, conf=%.2f); "
                    "distance metrics may be unreliable",
                    self._active_calibration_source,
                    self._active_calibration_confidence,
                )
            self._shot_locked_ppm = current_ppm
            self.tracker.set_calibration(self._shot_locked_ppm)
            logger.info(
                "Calibration locked at TRACKING start: %.1f px/m (source=%s, conf=%.2f)",
                self._shot_locked_ppm,
                self._active_calibration_source,
                self._active_calibration_confidence,
            )
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
        
        # Notify sensor fusion on shot lifecycle transitions
        if current_state == ShotState.TRACKING and self._previous_game_state != ShotState.TRACKING:
            self.sensor_fusion.on_shot_start(frame_data.timestamp_ns)
            self.sensor_fusion.set_calibration(
                self.get_pixels_per_meter(),
                getattr(get_config().calibration, 'forward_direction_deg', 0.0),
            )
        
        # Analyze shot when transitioning to STOPPED
        if (current_state == ShotState.STOPPED and 
            self._previous_game_state != ShotState.STOPPED and
            not self._shot_analyzed):
            self._analyze_completed_shot()
            self._shot_analyzed = True
        
        # Reset shot analyzed flag and depth trackers when returning to ARMED
        if current_state == ShotState.ARMED:
            self._shot_analyzed = False
            if self._previous_game_state == ShotState.COOLDOWN:
                self.ball_tracker_3d.reset()
                self.club_tracker.reset()
                self.launch_detector.reset()
        
        # Track previous state for transition detection
        self._previous_game_state = current_state
    
    def _analyze_completed_shot(self):
        """Analyze a completed shot and record in session."""
        if not self._current_state or not self._current_state.shot_result:
            return
        
        shot_result = self._current_state.shot_result
        ppm = self._shot_locked_ppm or self.get_pixels_per_meter()
        
        # Build shot data dict for analysis
        shot_data = {
            'speed_m_s': shot_result.initial_speed_px_s / ppm,
            'distance_m': shot_result.total_distance_px / ppm,
            'direction_deg': shot_result.initial_direction_deg,
        }
        
        # Check if drill is active
        drill_state = self.drill_manager.get_state()
        if drill_state.get('active'):
            # Record drill attempt
            self.drill_manager.record_attempt(
                actual_distance_m=shot_data['distance_m'],
                direction_deg=shot_data['direction_deg']
            )
            logger.info(f"Drill attempt recorded: distance={shot_data['distance_m']:.2f}m")
            self._build_fused_shot_report(shot_data)
            return  # Don't also record as regular session shot during drills
        
        # Analyze the shot against the hole
        analysis = self.game_logic.analyze_shot_from_backend_state(shot_data, ppm)
        
        if analysis:
            # Record in session
            self.session_manager.record_shot(
                analysis=analysis,
                shot_data=shot_data,
                target_distance_m=self.game_logic.get_hole_distance()
            )
            
            logger.info(
                f"Shot analyzed: {analysis.result.value}, "
                f"made={self.session_manager.get_putts_made()}/{self.session_manager.get_total_putts()} "
                f"({self.session_manager.get_make_percentage():.1f}%)"
            )
        
        # Build fused shot report from multi-camera data
        self._build_fused_shot_report(shot_data)
    
    def _build_fused_shot_report(self, shot_data: dict):
        """Build a fused shot report combining all camera data."""
        try:
            mc = get_config().multi_camera
            fused = self.sensor_fusion.build_shot_report(
                ball_speed_m_s=shot_data['speed_m_s'],
                distance_m=shot_data['distance_m'],
                direction_deg=shot_data['direction_deg'],
                shot_timestamp_ns=int(time.time_ns()),
                require_all_cameras_for_official=getattr(
                    mc, "require_all_cameras_for_recorded_putts", False
                ),
            )
            
            trajectory = []
            if self._current_state and self._current_state.shot_result:
                trajectory = self._current_state.shot_result.trajectory
            
            if fused.shot_type == "chip":
                self._latest_shot_report = self.chipping_analyzer.analyze(
                    fused, trajectory,
                    self.ball_tracker_3d.get_trajectory(),
                    self.sensor_fusion.shot_ball_states,
                    self.get_pixels_per_meter(),
                )
            else:
                self._latest_shot_report = self.putting_analyzer.analyze(
                    fused, trajectory, self.get_pixels_per_meter(),
                )
            
            # Propagate fast-putt flags from tracker into fused report
            if self._current_state and self._current_state.shot_result:
                sr = self._current_state.shot_result
                fused.fast_putt_estimated = sr.fast_putt_estimated
                if sr.fast_putt_estimated:
                    fused.fast_putt_resolved = False

            self._latest_fused_report = fused
            
            cameras = ', '.join(fused.cameras_used)
            logger.info(
                "Fused shot report: type=%s, cameras=[%s], primary=%s, fusion_accepted=%s, reject_reason=%s",
                fused.shot_type,
                cameras,
                fused.primary_source,
                fused.fusion_accepted,
                fused.fusion_rejected_reason,
            )
            
            # Reset depth trackers for next shot
            self.ball_tracker_3d.reset()
            self.club_tracker.reset()
            self.launch_detector.reset()
            self.sensor_fusion.on_shot_end()
        except Exception as e:
            logger.error(f"Error building fused shot report: {e}")
    
    def _process_zed_frame(self, frame):
        """Process a frame from the ZED 2i camera (runs in ZED thread)."""
        try:
            self._zed_frame = frame.color
            
            # Skip processing if no depth available
            if frame.depth is None:
                return
            
            current_state = self._current_state
            ball_in_motion = (current_state is not None and
                              current_state.state in (ShotState.TRACKING, ShotState.VIRTUAL_ROLLING))
            
            # 3D ball tracking
            ball_3d = self.ball_tracker_3d.update(
                frame.color, frame.depth, frame.point_cloud, frame.timestamp_ns,
            )
            if ball_3d.position:
                self._last_zed_detection_monotonic_ns = time.monotonic_ns()
                self.sensor_fusion.feed_ball_3d(ball_3d)
                # Update club tracker with ball position
                self.club_tracker.set_ball_position(
                    ball_3d.position.px_x, ball_3d.position.px_y,
                    (ball_3d.position.x, ball_3d.position.y, ball_3d.position.z),
                )
            
            # Club tracking
            club_metrics = self.club_tracker.update(
                frame.color, frame.depth, frame.point_cloud,
                frame.timestamp_ns, ball_in_motion,
            )
            if club_metrics:
                self.sensor_fusion.feed_club_metrics(club_metrics)
                
        except Exception as e:
            logger.debug(f"ZED processing error: {e}")
    
    def _process_realsense_frame(self, frame):
        """Process a frame from the RealSense D455 camera (runs in RS thread)."""
        try:
            self._rs_frame = frame.color
            
            if frame.depth is None:
                return
            
            current_state = self._current_state
            ball_in_motion = (current_state is not None and
                              current_state.state in (ShotState.TRACKING, ShotState.VIRTUAL_ROLLING))
            
            launch_data = self.launch_detector.update(
                frame.color, frame.depth, frame.timestamp_ns, ball_in_motion,
            )
            if launch_data:
                self._last_realsense_detection_monotonic_ns = time.monotonic_ns()
                self.sensor_fusion.feed_launch_data(launch_data)
                
        except Exception as e:
            logger.debug(f"RealSense processing error: {e}")
    
    def get_pixels_per_meter(self) -> float:
        """
        Single source of truth for calibration scale.
        Priority: high-confidence auto > manual override > homography > config/default.
        """
        config = get_config()

        auto_ppm = self.auto_calibrator.pixels_per_meter if self.auto_calibrator.is_calibrated else 0.0
        auto_conf = self.auto_calibrator.confidence if self.auto_calibrator.is_calibrated else 0.0
        manual_ppm = float(getattr(config.calibration, "manual_pixels_per_meter", 0.0) or 0.0)

        if auto_ppm > 0 and auto_conf >= 0.88:
            self._active_calibration_ppm = auto_ppm
            self._active_calibration_confidence = auto_conf
            self._active_calibration_source = "auto"
            if manual_ppm > 0 and abs(auto_ppm - manual_ppm) / auto_ppm > 0.08:
                if not getattr(self, '_stale_manual_calib_warned', False):
                    self._stale_manual_calib_warned = True
                    logger.warning(
                        "Manual calibration appears stale and is ignored this run: manual=%.1f px/m, auto=%.1f px/m. "
                        "Consider clearing manual_pixels_per_meter from config.json.",
                        manual_ppm,
                        auto_ppm,
                    )
        elif manual_ppm > 0:
            self._active_calibration_ppm = manual_ppm
            self._active_calibration_confidence = 0.95
            self._active_calibration_source = "manual"
        elif self.calibrator.is_calibrated and self.calibrator.pixels_per_meter > 0:
            self._active_calibration_ppm = self.calibrator.pixels_per_meter
            self._active_calibration_confidence = 0.7
            self._active_calibration_source = "homography"
        elif config.calibration.pixels_per_meter > 0:
            self._active_calibration_ppm = float(config.calibration.pixels_per_meter)
            self._active_calibration_confidence = 0.5
            self._active_calibration_source = "config"
        else:
            self._active_calibration_ppm = 1150.0
            self._active_calibration_confidence = 0.0
            self._active_calibration_source = "fallback"

        scale_factor = getattr(config.calibration, 'distance_scale_factor', 1.0)
        if scale_factor and scale_factor != 1.0:
            return self._active_calibration_ppm / scale_factor
        return self._active_calibration_ppm
    
    def get_state_message(self) -> dict:
        """Build state message for WebSocket."""
        state = self._current_state
        self._ws_broadcast_seq += 1
        mc = get_config().multi_camera
        _n = max(1, int(getattr(mc, "ws_broadcast_lightweight_every_n", 2)))
        lightweight = (self._ws_broadcast_seq % _n) != 0
        traj_limit = 8 if lightweight else 50
        
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
                "trajectory_before_exit": es.trajectory_before_exit[-traj_limit:],
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
                    pred_lim = 12 if lightweight else 50
                    prediction_data = {
                        "trajectory": [(p.x, p.y) for p in prediction.trajectory[:pred_lim]],
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
                "trajectory": result.trajectory[-traj_limit:],
                "exited_frame": bool(result.exited_frame),
                "fast_putt_estimated": bool(result.fast_putt_estimated),
                "shot_confidence": round(float(getattr(result, "shot_confidence", 0.0)), 3),
            }
        
        # Metrics (cap_fps tick once; passed into multi_camera for Arducam tile)
        cap_fps_val = round(self._fps_snapshot(self._cap_fps), 1)
        proc_fps_val = round(self._fps_snapshot(self._proc_fps), 1)
        backlog = max(0, self._arducam_capture_count - self._arducam_processed_count)
        metrics = {
            "cap_fps": cap_fps_val,
            "proc_fps": proc_fps_val,
            "disp_fps": round(self._disp_fps.tick(), 1),
            "proc_latency_ms": round(self._last_proc_time, 2),
            "frame_age_ms": round(self._last_proc_frame_age_ms, 2),
            "frame_age_avg_ms": round(self._proc_frame_age_ema_ms, 2),
            "queue_backlog": backlog,
            "dropped_frames": self._arducam_dropped_count,
            "idle_stddev": round(state.idle_stddev, 2) if state else 0.0,
            "startup_phase": self._startup_phase,
            "startup_arducam_profile": self._startup_arducam_profile_info,
            "startup_sustained_fps": round(self._startup_sustained_fps, 1),
            "startup_fail_reason": self._startup_fail_reason,
        }
        
        # Get overlay_radius_scale from config (for UI display only)
        config = get_config()
        overlay_radius_scale = getattr(config.calibration, 'overlay_radius_scale', 1.15)
        
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
            "calibration_source": self._active_calibration_source,
            "calibration_confidence": round(self._active_calibration_confidence, 3),
            "overlay_radius_scale": overlay_radius_scale,
            "resolution": list(self.camera.resolution) if self.camera else [1280, 800],
            "ready_status": state.ready_status if state else "no_ball",
            # Game logic and session data
            "game": self.game_logic.get_state_for_websocket(),
            "session": self.session_manager.get_state_for_websocket(),
            "drill": self.drill_manager.get_state(),
            # Multi-camera data
            "multi_camera": self._get_multi_camera_state(arducam_fps=cap_fps_val),
        }
    
    def _get_multi_camera_state(self, arducam_fps: float = 0.0) -> dict:
        """Build multi-camera state for WebSocket."""
        cameras: Dict[str, Any] = {}
        # Primary Arducam (legacy Camera path — not CameraManager ArducamSource)
        age_ms = None
        if self._last_arducam_frame_monotonic_ns is not None:
            age_ms = (time.monotonic_ns() - self._last_arducam_frame_monotonic_ns) / 1e6
        cameras["arducam"] = {
            "type": "arducam",
            "connected": bool(self.camera),
            "running": bool(self.camera and self.camera.is_running),
            "fps": round(arducam_fps, 1),
            "resolution": list(self.camera.resolution) if self.camera else [0, 0],
            "error": self._startup_fail_reason,
            "frame_count": self._frame_id,
            "target_fps": round(self._arducam_target_fps, 1),
            "min_healthy_fps": round(self._arducam_target_fps * ARDUCAM_MIN_HEALTHY_FPS_RATIO, 1),
            "consecutive_read_failures": self._arducam_consecutive_read_failures,
            "capture_count": self._arducam_capture_count,
            "processed_count": self._arducam_processed_count,
            "dropped_frames": self._arducam_dropped_count,
            "queue_backlog": max(0, self._arducam_capture_count - self._arducam_processed_count),
            "tracker_contribution": True,
            "startup_profile": self._startup_arducam_profile_info,
            "sustained_startup_fps": round(self._startup_sustained_fps, 1),
        }
        if age_ms is not None:
            cameras["arducam"]["last_frame_age_ms"] = round(age_ms, 2)
        if self._last_arducam_detection_monotonic_ns is not None:
            cameras["arducam"]["last_detection_age_ms"] = round(
                (time.monotonic_ns() - self._last_arducam_detection_monotonic_ns) / 1e6,
                2,
            )
        if self.camera:
            cameras["arducam"]["driver_reported_fps"] = round(self.camera.reported_fps, 1)
        for ct in (CameraType.ZED, CameraType.REALSENSE):
            src = self.camera_manager.get_source(ct)
            if src:
                cameras[ct.value] = src.status().to_dict()
                if ct == CameraType.ZED:
                    cameras[ct.value]["tracker_contribution"] = bool(self._last_zed_detection_monotonic_ns is not None)
                    if self._last_zed_detection_monotonic_ns is not None:
                        cameras[ct.value]["last_detection_age_ms"] = round(
                            (time.monotonic_ns() - self._last_zed_detection_monotonic_ns) / 1e6,
                            2,
                        )
                else:
                    cameras[ct.value]["tracker_contribution"] = bool(self._last_realsense_detection_monotonic_ns is not None)
                    if self._last_realsense_detection_monotonic_ns is not None:
                        cameras[ct.value]["last_detection_age_ms"] = round(
                            (time.monotonic_ns() - self._last_realsense_detection_monotonic_ns) / 1e6,
                            2,
                        )
        
        # Latest fused shot report
        shot_report = None
        if self._latest_shot_report:
            shot_report = self._latest_shot_report.to_dict()
        
        # Club tracker state
        club_state = {
            "stroke_phase": self.club_tracker.stroke_phase.value,
        }
        latest_club = self.club_tracker.get_latest_metrics()
        if latest_club:
            club_state["metrics"] = latest_club.to_dict()
        
        # Rollup: all three present and recently received frames
        want = ("arducam", "zed", "realsense")
        all_ok = all(
            k in cameras and bool(cameras[k].get("connected")) and bool(cameras[k].get("running"))
            for k in want
        )
        stale = False
        max_age = 0.0
        for k in want:
            st = cameras.get(k) or {}
            a = st.get("last_frame_age_ms")
            if a is not None:
                max_age = max(max_age, float(a))
                if float(a) > 500.0:
                    stale = True
            elif st.get("running"):
                # Running camera without frame-age telemetry yet should not be treated as healthy.
                stale = True
        system_health = {
            "all_streams_reporting": all_ok,
            "max_last_frame_age_ms": round(max_age, 2) if max_age else None,
            "stale_warning": stale,
        }

        current_state = self._current_state
        game_state = current_state.state.value if current_state else "armed"
        tracking_activity = {
            "game_state": game_state,
            "arducam_ball_tracking": bool(
                current_state
                and current_state.ball_x is not None
                and current_state.state in (ShotState.TRACKING, ShotState.VIRTUAL_ROLLING)
            ),
            "zed_club_phase": self.club_tracker.stroke_phase.value,
            "realsense_launch_active": bool(getattr(self.launch_detector, "_tracking_active", False)),
        }

        return {
            "cameras": cameras,
            "shot_report": shot_report,
            "club": club_state,
            "system_health": system_health,
            "tracking_activity": tracking_activity,
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

# Add CORS middleware to allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
if not frontend_path.exists():
    # Fallback to legacy or raw frontend if dist doesn't exist
    frontend_path = Path(__file__).parent.parent / "frontend_legacy"

if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
else:
    # Try the new source folder just in case (though it won't work well without build)
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/")
async def root():
    """Serve main page."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>StrikeLab Putting Sim</h1><p>Frontend not found. Run 'npm run build' in frontend/</p>")


@app.get("/api/video")
async def video_feed():
    """MJPEG video stream for viewing camera output (low fps to avoid starving tracker)."""
    
    def generate_frames():
        import cv2
        sim = get_app_instance()
        last_frame_time = 0.0
        last_frame_ref = None
        TARGET_FPS = 15
        
        while sim._running:
            now = time.time()
            dt = now - last_frame_time
            if dt < 1 / TARGET_FPS:
                time.sleep(max(0.001, 1 / TARGET_FPS - dt))
                continue
            last_frame_time = now
            
            ref = sim._current_frame
            if ref is not None:
                last_frame_ref = ref
                frame = ref.copy()
                
                state = sim._current_state
                if state and state.ball_x is not None:
                    cx, cy = round(state.ball_x), round(state.ball_y)
                    radius = int(state.ball_radius or 15)
                    
                    color = (0, 255, 0)
                    if state.state.value == "TRACKING":
                        color = (0, 0, 255)
                    elif state.state.value == "STOPPED":
                        color = (0, 255, 255)
                    
                    cv2.circle(frame, (cx, cy), radius, color, 2)
                    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 1)
                    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 1)
                    
                    cv2.putText(frame, state.state.value, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    if state.velocity and state.state.value == "TRACKING":
                        speed_ms = state.velocity.speed / sim.calibrator.pixels_per_meter
                        cv2.putText(frame, f"{speed_ms:.2f} m/s", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@app.get("/api/video/zed")
async def zed_video_feed():
    """MJPEG video stream from the ZED 2i left camera."""
    def generate_frames():
        sim = get_app_instance()
        last_t = 0.0
        last_frame_ref = None
        TARGET_FPS = 10
        while sim._running:
            now = time.time()
            dt = now - last_t
            if dt < 1 / TARGET_FPS:
                time.sleep(max(0.001, 1 / TARGET_FPS - dt))
                continue
            last_t = now
            ref = sim._zed_frame
            if ref is not None and ref is not last_frame_ref:
                last_frame_ref = ref
                frame = ref.copy()
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            elif ref is None:
                time.sleep(0.1)

    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'},
    )


@app.get("/api/video/realsense")
async def realsense_video_feed():
    """MJPEG video stream from the RealSense D455 color camera."""
    def generate_frames():
        sim = get_app_instance()
        last_t = 0.0
        last_frame_ref = None
        TARGET_FPS = 10
        while sim._running:
            now = time.time()
            dt = now - last_t
            if dt < 1 / TARGET_FPS:
                time.sleep(max(0.001, 1 / TARGET_FPS - dt))
                continue
            last_t = now
            ref = sim._rs_frame
            if ref is not None and ref is not last_frame_ref:
                last_frame_ref = ref
                frame = ref.copy()
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            elif ref is None:
                time.sleep(0.1)

    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'},
    )


@app.get("/api/cameras/status")
async def get_cameras_status():
    """Get status of all cameras."""
    try:
        sim = get_app_instance()
        statuses = {}
        for ct in (CameraType.ZED, CameraType.REALSENSE):
            src = sim.camera_manager.get_source(ct)
            if src:
                statuses[ct.value] = src.status().to_dict()
        statuses["arducam"] = {
            "type": "arducam",
            "connected": sim.camera is not None and sim.camera.is_running,
            "running": sim._running,
            "fps": round(sim.camera.fps, 1) if sim.camera else 0,
            "resolution": list(sim.camera.resolution) if sim.camera else [0, 0],
        }
        return JSONResponse({"cameras": statuses})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v1/shot/latest")
async def get_latest_shot_report():
    """Get the latest full shot report with all TrackMan-style metrics."""
    try:
        sim = get_app_instance()
        if sim._latest_shot_report:
            return JSONResponse(sim._latest_shot_report.to_dict())
        return JSONResponse({"error": "No shot recorded yet"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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


@app.get("/api/health")
async def health_check():
    """
    Readiness probe used by the top-level launcher to gate frontend startup.
    Returns structured readiness with camera health so the launcher can
    distinguish server_up / camera_ready / degraded states.
    """
    try:
        sim = get_app_instance()
    except RuntimeError:
        return JSONResponse({
            "status": "starting",
            "server_up": True,
            "camera_ready": False,
            "startup_phase": "init",
            "arducam": None,
            "degraded_reasons": ["app_not_initialized"],
        })

    camera_ready = sim.camera is not None and sim._running
    arducam_fps = PuttingSimApp._fps_snapshot(sim._cap_fps)
    degraded_reasons: list[str] = []
    if sim._startup_fail_reason:
        degraded_reasons.append(sim._startup_fail_reason)
    if camera_ready and arducam_fps > 0 and arducam_fps < 20:
        degraded_reasons.append(f"arducam_low_fps:{arducam_fps:.1f}")

    phase = sim._startup_phase
    if phase == "running" and camera_ready:
        status = "degraded" if degraded_reasons else "ready"
    elif phase == "error":
        status = "error"
    else:
        status = "starting"

    resolution = list(sim.camera.resolution) if sim.camera else [0, 0]
    return JSONResponse({
        "status": status,
        "server_up": True,
        "camera_ready": camera_ready,
        "startup_phase": phase,
        "arducam": {
            "connected": sim.camera is not None,
            "resolution": resolution,
            "fps": round(arducam_fps, 1),
            "sustained_startup_fps": round(sim._startup_sustained_fps, 1),
            "profile": sim._startup_arducam_profile_info,
        },
        "degraded_reasons": degraded_reasons,
    })


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
    """Reset tracker state, shot report, game analysis, and fusion caches."""
    try:
        sim = get_app_instance()
        sim.reset_all()
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============= Game Logic Endpoints =============

@app.get("/api/game/hole")
async def get_hole_config():
    """Get current hole configuration."""
    try:
        sim = get_app_instance()
        return JSONResponse({
            "success": True,
            "hole_distance_m": sim.game_logic.get_hole_distance(),
            "hole_radius_m": sim.game_logic.hole.radius_m
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/game/hole")
async def set_hole_distance(data: dict):
    """
    Set the hole distance.
    
    Expected data:
    {
        "distance_m": float  # Distance in meters (1-10)
    }
    """
    try:
        sim = get_app_instance()
        distance_m = data.get("distance_m")
        
        if distance_m is None:
            return JSONResponse({
                "success": False,
                "error": "distance_m is required"
            }, status_code=400)
        
        # Validate range
        if not 0.5 <= distance_m <= 25.0:
            return JSONResponse({
                "success": False,
                "error": "distance_m must be between 0.5 and 25.0 meters"
            }, status_code=400)
        
        sim.game_logic.set_hole_distance(distance_m)
        
        return JSONResponse({
            "success": True,
            "hole_distance_m": distance_m
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/game/last-shot")
async def get_last_shot_result():
    """Get the result of the last analyzed shot."""
    try:
        sim = get_app_instance()
        analysis = sim.game_logic.get_last_analysis()
        
        if not analysis:
            return JSONResponse({
                "success": False,
                "error": "No shot has been analyzed yet"
            })
        
        return JSONResponse({
            "success": True,
            "result": analysis.result.value,
            "is_made": analysis.is_made,
            "distance_to_hole_m": round(analysis.distance_to_hole_m, 3),
            "lateral_miss_m": round(analysis.lateral_miss_m, 3),
            "depth_miss_m": round(analysis.depth_miss_m, 3),
            "miss_description": analysis.miss_description
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============= User Management Endpoints =============

@app.get("/api/users")
async def get_users():
    """Get all users."""
    try:
        from .database import get_database
        db = get_database()
        users = db.get_users()
        return JSONResponse({
            "success": True,
            "users": [asdict(u) for u in users]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/users")
async def create_user(data: dict):
    """
    Create a new user.
    
    Expected data:
    {
        "name": str,
        "handicap": float (optional, default 0.0)
    }
    """
    try:
        from .database import get_database
        db = get_database()
        
        name = data.get("name")
        handicap = data.get("handicap", 0.0)
        
        if not name:
            return JSONResponse({
                "success": False,
                "error": "name is required"
            }, status_code=400)
            
        user_id = db.create_user(name, handicap)
        
        return JSONResponse({
            "success": True,
            "user": {
                "id": user_id,
                "name": name,
                "handicap": handicap
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user."""
    try:
        from .database import get_database
        db = get_database()
        
        db.delete_user(user_id)
        
        # If current session user is this user, clear it
        sim = get_app_instance()
        if sim.session_manager.current_user_id == user_id:
            sim.session_manager.set_user(None)
            
        return JSONResponse({
            "success": True,
            "message": f"User {user_id} deleted"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/users/{user_id}/reset")
async def reset_user_data(user_id: int):
    """
    Reset all data for a user without deleting the user account.
    This clears all shots and sessions for the user.
    """
    try:
        from .database import get_database
        db = get_database()
        
        # Verify user exists
        users = db.get_users()
        user_exists = any(u.id == user_id for u in users)
        if not user_exists:
            return JSONResponse({
                "success": False,
                "error": f"User {user_id} not found"
            }, status_code=404)
        
        # Reset the data
        result = db.reset_user_data(user_id)
        
        # If current session user is this user, reset the session too
        sim = get_app_instance()
        if sim.session_manager.current_user_id == user_id:
            sim.session_manager.reset()
            
        return JSONResponse({
            "success": True,
            "message": f"Data reset for user {user_id}",
            "shots_deleted": result["shots_deleted"],
            "sessions_deleted": result["sessions_deleted"]
        })
    except Exception as e:
        logger.error(f"Failed to reset user data: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============= Session Endpoints =============

@app.get("/api/session")
async def get_session():
    """Get current session statistics."""
    try:
        sim = get_app_instance()
        return JSONResponse({
            "success": True,
            **sim.session_manager.get_state_for_websocket()
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/session/user")
async def set_session_user(data: dict):
    """
    Set the active user for the session.
    
    Expected data:
    {
        "user_id": int | null
    }
    """
    try:
        sim = get_app_instance()
        user_id = data.get("user_id")
        
        sim.session_manager.set_user(user_id)
        
        return JSONResponse({
            "success": True,
            "user_id": user_id,
            **sim.session_manager.get_state_for_websocket()
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/session/reset")
async def reset_session():
    """Reset the current session and start a new one."""
    try:
        sim = get_app_instance()
        sim.session_manager.reset()
        return JSONResponse({
            "success": True,
            "message": "Session reset",
            **sim.session_manager.get_state_for_websocket()
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============= Drill Endpoints =============

@app.get("/api/drill")
async def get_drill_state():
    """Get current drill state."""
    try:
        sim = get_app_instance()
        return JSONResponse({
            "success": True,
            **sim.drill_manager.get_state()
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/drill/start")
async def start_drill(data: dict):
    """
    Start a drill session.
    
    Expected data:
    {
        "drill_type": "distance_control" | "ladder_drill"
    }
    """
    try:
        sim = get_app_instance()
        drill_type_str = data.get("drill_type", "distance_control")
        
        try:
            drill_type = DrillType(drill_type_str)
        except ValueError:
            return JSONResponse({
                "success": False,
                "error": f"Invalid drill type: {drill_type_str}. Valid types: distance_control, ladder_drill"
            }, status_code=400)
        
        state = sim.drill_manager.start_drill(drill_type)
        
        return JSONResponse({
            "success": True,
            "message": f"Started {drill_type.value} drill",
            **state
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/drill/stop")
async def stop_drill():
    """Stop the current drill session."""
    try:
        sim = get_app_instance()
        state = sim.drill_manager.stop_drill()
        
        return JSONResponse({
            "success": True,
            "message": "Drill stopped",
            "final_state": state
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/session/history")
async def get_session_history():
    """Get shot history for the current session."""
    try:
        sim = get_app_instance()
        shots = sim.session_manager.get_last_n_shots(50)
        
        return JSONResponse({
            "success": True,
            "session_id": sim.session_manager.session_id,
            "total_shots": sim.session_manager.get_total_putts(),
            "shots": [
                {
                    "shot_number": idx + 1,  # 1-based shot number for this session
                    "id": shot.id,  # Database ID
                    "timestamp": shot.timestamp,
                    "speed_m_s": round(shot.speed_m_s, 3),
                    "distance_m": round(shot.distance_m, 3),
                    "direction_deg": round(shot.direction_deg, 2),
                    "target_distance_m": round(shot.target_distance_m, 2),
                    "result": shot.result.value,
                    "is_made": shot.is_made,
                    "distance_to_hole_m": round(shot.distance_to_hole_m, 3)
                }
                for idx, shot in enumerate(shots)
            ]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/users/{user_id}/history")
async def get_user_shot_history(user_id: int, limit: int = 100):
    """
    Get shot history for a specific user from the database.
    This returns all historical shots, not just the current session.
    """
    try:
        from .database import get_database
        db = get_database()
        
        # Verify user exists
        users = db.get_users()
        user = next((u for u in users if u.id == user_id), None)
        if not user:
            return JSONResponse({
                "success": False,
                "error": f"User {user_id} not found"
            }, status_code=404)
        
        # Get shots from database
        shots = db.get_shots(user_id=user_id, limit=limit)
        
        # Group shots by session for shot numbering
        shots_by_session: dict = {}
        for shot in reversed(shots):  # Reverse to process chronologically
            session_id = shot.session_id
            if session_id not in shots_by_session:
                shots_by_session[session_id] = []
            shots_by_session[session_id].append(shot)
        
        # Build response with session-based shot numbers
        result_shots = []
        for shot in shots:  # Already in reverse chronological order
            session_shots = shots_by_session.get(shot.session_id, [])
            # Find shot index within session (0-based)
            shot_idx = next((i for i, s in enumerate(session_shots) if s.id == shot.id), 0)
            
            result_shots.append({
                "shot_number": shot_idx + 1,  # 1-based within session
                "id": shot.id,
                "session_id": shot.session_id,
                "timestamp": shot.timestamp,
                "speed_m_s": round(shot.speed_m_s, 3),
                "distance_m": round(shot.distance_m, 3),
                "direction_deg": round(shot.direction_deg, 2),
                "target_distance_m": round(shot.target_distance_m, 2),
                "result": shot.result,
                "is_made": shot.is_made,
                "distance_to_hole_m": round(shot.distance_to_hole_m, 3) if shot.distance_to_hole_m else None
            })
        
        return JSONResponse({
            "success": True,
            "user_id": user_id,
            "user_name": user.name,
            "total_shots": len(shots),
            "sessions_count": len(shots_by_session),
            "shots": result_shots
        })
    except Exception as e:
        logger.error(f"Failed to get user shot history: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============= Green Speed Calibration Endpoints =============

# Green speed presets (deceleration in m/s²)
GREEN_SPEED_PRESETS = {
    "fast": {"deceleration": 0.45, "stimp": "12-14", "description": "Tournament fast green"},
    "medium-fast": {"deceleration": 0.55, "stimp": "10-12", "description": "Club championship"},
    "medium": {"deceleration": 0.65, "stimp": "8-10", "description": "Standard club green"},
    "medium-slow": {"deceleration": 0.80, "stimp": "6-8", "description": "Casual play"},
    "slow": {"deceleration": 1.00, "stimp": "4-6", "description": "Practice mat / rough carpet"},
}


@app.get("/api/green-speed")
async def get_green_speed():
    """Get current green speed settings."""
    try:
        config = get_config()
        current_decel = config.calibration.virtual_deceleration_m_s2
        
        # Find matching preset
        current_preset = None
        for name, preset in GREEN_SPEED_PRESETS.items():
            if abs(preset["deceleration"] - current_decel) < 0.02:
                current_preset = name
                break
        
        return JSONResponse({
            "success": True,
            "current_deceleration_m_s2": current_decel,
            "current_preset": current_preset,
            "presets": GREEN_SPEED_PRESETS
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/green-speed")
async def set_green_speed(data: dict):
    """
    Set green speed using a preset or custom value.
    
    Expected data (one of):
    {
        "preset": "fast" | "medium-fast" | "medium" | "medium-slow" | "slow"
    }
    OR
    {
        "deceleration_m_s2": float  # Custom value between 0.3 and 1.5
    }
    """
    try:
        sim = get_app_instance()
        config_mgr = get_config_manager()
        
        preset = data.get("preset")
        custom_decel = data.get("deceleration_m_s2")
        
        if preset:
            if preset not in GREEN_SPEED_PRESETS:
                return JSONResponse({
                    "success": False,
                    "error": f"Invalid preset: {preset}. Valid presets: {list(GREEN_SPEED_PRESETS.keys())}"
                }, status_code=400)
            
            decel = GREEN_SPEED_PRESETS[preset]["deceleration"]
            logger.info(f"Setting green speed to preset '{preset}' (deceleration={decel} m/s²)")
        
        elif custom_decel is not None:
            if not 0.3 <= custom_decel <= 1.5:
                return JSONResponse({
                    "success": False,
                    "error": "deceleration_m_s2 must be between 0.3 and 1.5"
                }, status_code=400)
            
            decel = custom_decel
            preset = "custom"
            logger.info(f"Setting custom green speed (deceleration={decel} m/s²)")
        
        else:
            return JSONResponse({
                "success": False,
                "error": "Must provide either 'preset' or 'deceleration_m_s2'"
            }, status_code=400)
        
        # Update config
        config_mgr.config.calibration.virtual_deceleration_m_s2 = decel
        config_mgr.save()
        
        # Update tracker
        sim.tracker.set_deceleration_m_s2(decel)
        
        return JSONResponse({
            "success": True,
            "preset": preset,
            "deceleration_m_s2": decel
        })
        
    except Exception as e:
        logger.error(f"Failed to set green speed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============= Database/Analytics Endpoints =============

@app.delete("/api/shots/{shot_id}")
async def delete_shot(shot_id: int):
    """Delete a specific shot."""
    try:
        from .database import get_database
        db = get_database()
        
        success = db.delete_shot(shot_id)
        
        if success:
            return JSONResponse({
                "success": True,
                "message": f"Shot {shot_id} deleted"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Shot not found"
            }, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stats/all-time")
async def get_all_time_stats():
    """Get all-time putting statistics from the database."""
    try:
        from .database import get_database
        db = get_database()
        
        stats = db.get_stats(days=0)  # 0 = all time
        return JSONResponse({
            "success": True,
            **stats
        })
    except Exception as e:
        logger.error(f"Failed to get all-time stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/users/{user_id}/stats")
async def get_user_stats(user_id: int):
    """
    Get all-time putting statistics for a specific user from the database.
    This includes historical data across all sessions.
    """
    try:
        from .database import get_database
        db = get_database()
        
        # Verify user exists
        users = db.get_users()
        user = next((u for u in users if u.id == user_id), None)
        if not user:
            return JSONResponse({
                "success": False,
                "error": f"User {user_id} not found"
            }, status_code=404)
        
        # Get user stats from database
        stats = db.get_stats(days=0, user_id=user_id)  # All time for this user
        shots = db.get_shots(user_id=user_id, limit=1000)  # Get all shots for user
        trend = db.get_recent_trend(n_shots=50, user_id=user_id)
        
        return JSONResponse({
            "success": True,
            "user": {
                "id": user.id,
                "name": user.name,
                "handicap": user.handicap,
                "created_at": user.created_at
            },
            "stats": stats,
            "total_shots_in_db": len(shots),
            "trend": trend
        })
    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stats/recent")
async def get_recent_stats():
    """Get statistics from the last 7 days."""
    try:
        from .database import get_database
        db = get_database()
        
        stats = db.get_stats(days=7)
        return JSONResponse({
            "success": True,
            **stats
        })
    except Exception as e:
        logger.error(f"Failed to get recent stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stats/trend")
async def get_trend_data():
    """Get trend data for the last 50 shots."""
    try:
        from .database import get_database
        db = get_database()
        
        trend = db.get_recent_trend(n_shots=50)
        return JSONResponse({
            "success": True,
            **trend
        })
    except Exception as e:
        logger.error(f"Failed to get trend data: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stats/by-distance")
async def get_stats_by_distance():
    """Get make percentage by distance band."""
    try:
        from .database import get_database
        db = get_database()
        
        stats = db.get_stats(days=0)
        return JSONResponse({
            "success": True,
            "by_distance": stats.get('by_distance', {})
        })
    except Exception as e:
        logger.error(f"Failed to get stats by distance: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stats/consistency")
async def get_consistency_stats():
    """
    Get detailed consistency and analytics for the current session.
    
    Returns:
    - Consistency metrics (stddev of speed, direction, distance)
    - Tendency analysis (bias detection)
    - Miss distribution (quadrant analysis)
    - Roll quality data (if available)
    """
    try:
        sim = get_app_instance()
        stats = sim.session_manager.get_stats()
        
        return JSONResponse({
            "success": True,
            "total_putts": stats.total_putts,
            "putts_made": stats.putts_made,
            "consistency": {
                "speed_stddev": round(stats.consistency.speed_stddev, 3),
                "direction_stddev": round(stats.consistency.direction_stddev, 2),
                "distance_error_stddev": round(stats.consistency.distance_error_stddev, 3),
                "speed_cv": round(stats.consistency.speed_cv, 1),
                "consistency_score": round(stats.consistency.consistency_score, 1),
                "rolling_speed_stddev": round(stats.consistency.rolling_speed_stddev, 3),
                "rolling_direction_stddev": round(stats.consistency.rolling_direction_stddev, 2),
                "rolling_window": stats.consistency.rolling_window,
            },
            "tendencies": {
                "speed_bias_m_s": round(stats.tendencies.speed_bias_m_s, 3),
                "distance_bias_m": round(stats.tendencies.distance_bias_m, 3),
                "direction_bias_deg": round(stats.tendencies.direction_bias_deg, 2),
                "lateral_bias_m": round(stats.tendencies.lateral_bias_m, 3),
                "dominant_miss": stats.tendencies.dominant_miss,
                "dominant_miss_percentage": round(stats.tendencies.dominant_miss_percentage, 1),
                "speed_tendency": stats.tendencies.speed_tendency,
                "direction_tendency": stats.tendencies.direction_tendency,
            },
            "miss_distribution": {
                "right_short": stats.miss_distribution.right_short,
                "right_long": stats.miss_distribution.right_long,
                "left_short": stats.miss_distribution.left_short,
                "left_long": stats.miss_distribution.left_long,
                "right_short_pct": stats.miss_distribution.right_short_pct,
                "right_long_pct": stats.miss_distribution.right_long_pct,
                "left_short_pct": stats.miss_distribution.left_short_pct,
                "left_long_pct": stats.miss_distribution.left_long_pct,
                "total_right": stats.miss_distribution.total_right,
                "total_left": stats.miss_distribution.total_left,
                "total_short": stats.miss_distribution.total_short,
                "total_long": stats.miss_distribution.total_long,
                "total_misses": stats.miss_distribution.total_misses,
            },
            "benchmarks": {
                "tour_speed_stddev": 0.1,  # Tour pros: ~0.1 m/s
                "tour_direction_stddev": 1.0,  # Tour pros: ~1°
                "tour_distance_error_stddev": 0.3,  # Tour pros: ~30cm
                "excellent_consistency_threshold": 80,
                "good_consistency_threshold": 60,
            }
        })
    except Exception as e:
        logger.error(f"Failed to get consistency stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/stats/export")
async def export_stats():
    """Export shot history to CSV."""
    try:
        from .database import get_database
        from pathlib import Path
        import tempfile
        
        db = get_database()
        
        # Create temp file for export
        export_path = Path(tempfile.gettempdir()) / "putting_export.csv"
        count = db.export_csv(export_path)
        
        if count == 0:
            return JSONResponse({
                "success": False,
                "error": "No shots to export"
            })
        
        return JSONResponse({
            "success": True,
            "exported_shots": count,
            "file_path": str(export_path)
        })
    except Exception as e:
        logger.error(f"Failed to export stats: {e}")
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
        
        # Get camera timing stats
        camera_fps_stats = None
        if sim.camera:
            stats = sim.camera.fps_stats
            camera_fps_stats = {
                "effective_fps": round(stats.fps, 1),
                "reported_fps": round(sim.camera.reported_fps, 1),
                "dt_mean_ms": round(stats.dt_mean_ms, 2),
                "dt_std_ms": round(stats.dt_std_ms, 2),
                "dt_min_ms": round(stats.dt_min_ms, 2),
                "dt_max_ms": round(stats.dt_max_ms, 2)
            }
        
        # Get shot timing stats from tracker
        shot_timing = None
        if sim.tracker.shot_timing_stats:
            ts = sim.tracker.shot_timing_stats
            shot_timing = {
                "effective_fps": round(ts.effective_fps, 1),
                "dt_mean_ms": round(ts.dt_mean_ms, 2),
                "dt_std_ms": round(ts.dt_std_ms, 2),
                "frame_count": ts.frame_count
            }
        
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
                "deceleration_px_s2": round(sim.tracker.get_deceleration_px_s2(), 1),
                "effective_fps": round(sim.tracker.effective_fps, 1)
            },
            "timing": {
                "camera": camera_fps_stats,
                "last_shot": shot_timing
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
                "initial_speed_px_s": float(round(result.initial_speed_px_s, 1)),
                "initial_speed_m_s": float(round(result.initial_speed_px_s / ppm, 3)),
                "direction_deg": float(round(result.initial_direction_deg, 2)),
                "physical_distance_px": float(round(result.total_distance_px - result.virtual_distance_px, 1)),
                "virtual_distance_px": float(round(result.virtual_distance_px, 1)),
                "total_distance_px": float(round(result.total_distance_px, 1)),
                "total_distance_m": float(round(result.total_distance_px / ppm, 4)),
                "total_distance_cm": float(round((result.total_distance_px / ppm) * 100, 1)),
                "exited_frame": bool(result.exited_frame),
                "duration_ms": float(round(result.duration_ms, 1)),
                "trajectory_points": int(len(result.trajectory))
            }
        
        return JSONResponse(diagnostics)
        
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/validation/report-shot")
async def report_validation_shot(data: dict):
    """
    Report a shot for validation purposes.
    
    Call this after each shot to record the ground truth distance for later analysis.
    
    Expected data:
    {
        "actual_distance_cm": float,  # Measured with ruler
        "notes": str                   # Optional notes
    }
    
    Returns comparison between system and ground truth.
    """
    try:
        sim = get_app_instance()
        state = sim._current_state
        
        actual_cm = data.get("actual_distance_cm")
        notes = data.get("notes", "")
        
        if actual_cm is None:
            return JSONResponse({
                "success": False,
                "error": "actual_distance_cm is required"
            }, status_code=400)
        
        if not state or not state.shot_result:
            return JSONResponse({
                "success": False,
                "error": "No shot recorded. Make a putt first."
            })
        
        result = state.shot_result
        ppm = result.pixels_per_meter if result.pixels_per_meter > 0 else sim.get_pixels_per_meter()
        
        # Get system measurements
        system_cm = (result.total_distance_px / ppm) * 100
        speed_m_s = result.initial_speed_px_s / ppm
        
        # Calculate errors
        error_cm = system_cm - actual_cm
        error_percent = (error_cm / actual_cm) * 100 if actual_cm > 0 else 0
        
        # Get timing stats
        shot_timing = None
        if sim.tracker.shot_timing_stats:
            ts = sim.tracker.shot_timing_stats
            shot_timing = {
                "effective_fps": round(ts.effective_fps, 1),
                "dt_mean_ms": round(ts.dt_mean_ms, 2),
                "frame_count": ts.frame_count
            }
        
        report = {
            "success": True,
            "ground_truth": {
                "actual_distance_cm": float(round(actual_cm, 1)),
                "notes": notes
            },
            "system_measurement": {
                "distance_cm": float(round(system_cm, 1)),
                "speed_m_s": float(round(speed_m_s, 3)),
                "direction_deg": float(round(result.initial_direction_deg, 2)),
                "physical_distance_cm": float(round((result.physical_distance_px / ppm) * 100, 1)),
                "virtual_distance_cm": float(round((result.virtual_distance_px / ppm) * 100, 1)),
                "exited_frame": bool(result.exited_frame),
                "pixels_per_meter": float(round(ppm, 1))
            },
            "error": {
                "error_cm": float(round(error_cm, 2)),
                "error_percent": float(round(error_percent, 2)),
                "within_2_percent": bool(abs(error_percent) <= 2.0),
                "within_5_percent": bool(abs(error_percent) <= 5.0)
            },
            "timing": shot_timing
        }
        
        # Log for later analysis
        logger.info(f"VALIDATION SHOT: actual={actual_cm:.1f}cm, system={system_cm:.1f}cm, "
                   f"error={error_cm:.1f}cm ({error_percent:.1f}%), exited={result.exited_frame}, "
                   f"speed={speed_m_s:.2f}m/s, notes='{notes}'")
        
        return JSONResponse(report)
        
    except Exception as e:
        logger.error(f"Validation report failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/calibration/static-ball-test")
async def static_ball_test():
    """
    Static ball test for calibration validation.
    
    Place the ball at a known position and call this endpoint to verify:
    - Detected radius vs expected radius
    - Pixels per meter calculation
    - Detection confidence
    
    Call this at 9 positions (center + 8 edges) for comprehensive testing.
    """
    try:
        sim = get_app_instance()
        state = sim._current_state
        
        if not state or state.ball_x is None:
            return JSONResponse({
                "success": False,
                "error": "No ball detected. Place ball in frame."
            })
        
        # Get current detection info
        ball_x = state.ball_x
        ball_y = state.ball_y
        ball_radius = state.ball_radius
        confidence = state.ball_confidence
        
        # Calculate expected values
        ppm = sim.get_pixels_per_meter()
        expected_radius = (0.04267 / 2) * ppm  # Ball diameter 42.67mm
        
        # Calculate radius error
        radius_error_px = ball_radius - expected_radius if ball_radius else 0
        radius_error_percent = (radius_error_px / expected_radius) * 100 if expected_radius > 0 else 0
        
        # Determine position in frame
        frame_w, frame_h = sim.camera.resolution if sim.camera else (1280, 800)
        position_name = "center"
        if ball_x < frame_w * 0.33:
            position_name = "left"
        elif ball_x > frame_w * 0.67:
            position_name = "right"
        if ball_y < frame_h * 0.33:
            position_name = "top-" + position_name if position_name != "center" else "top"
        elif ball_y > frame_h * 0.67:
            position_name = "bottom-" + position_name if position_name != "center" else "bottom"
        
        return JSONResponse({
            "success": True,
            "position": {
                "x_px": round(ball_x, 1),
                "y_px": round(ball_y, 1),
                "position_name": position_name
            },
            "radius": {
                "detected_px": round(ball_radius, 2) if ball_radius else None,
                "expected_px": round(expected_radius, 2),
                "error_px": round(radius_error_px, 2),
                "error_percent": round(radius_error_percent, 2)
            },
            "confidence": round(confidence, 3) if confidence else None,
            "calibration": {
                "pixels_per_meter": round(ppm, 1),
                "auto_calibrated": sim.auto_calibrator.is_calibrated,
                "lens_calibrated": sim._lens_calibrated
            },
            "overlay_recommendation": {
                "scale_factor": round(ball_radius / expected_radius, 3) if ball_radius and expected_radius > 0 else 1.0,
                "message": (
                    "Overlay matches ball size" if abs(radius_error_percent) < 5
                    else f"Overlay is {'smaller' if radius_error_percent < 0 else 'larger'} than ball - "
                         f"consider {'increasing' if radius_error_percent < 0 else 'decreasing'} overlay radius by {abs(radius_error_percent):.1f}%"
                )
            }
        })
        
    except Exception as e:
        logger.error(f"Static ball test failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/calibration/9-position-overlay-test")
async def nine_position_overlay_test():
    """
    9-position static overlay test for calibration validation.
    
    Test positions: center, top, bottom, left, right, 
                    top-left, top-right, bottom-left, bottom-right
    
    For each position, report:
    - detected_radius_px
    - expected_radius_px (from ball diameter + ppm)
    - recommended_overlay_scale
    - position_name
    
    Call this endpoint once with the ball at each position to build a complete report.
    This endpoint returns the result for the current ball position.
    """
    try:
        sim = get_app_instance()
        state = sim._current_state
        config = get_config()
        
        if not state or state.ball_x is None:
            return JSONResponse({
                "success": False,
                "error": "No ball detected. Place ball in frame."
            })
        
        # Get current detection info
        ball_x = state.ball_x
        ball_y = state.ball_y
        detected_radius = state.ball_radius
        confidence = state.ball_confidence
        
        # Calculate expected values
        ppm = sim.get_pixels_per_meter()
        BALL_DIAMETER_M = 0.04267  # Golf ball diameter 42.67mm
        expected_radius = (BALL_DIAMETER_M / 2) * ppm
        
        # Get current overlay scale from config
        current_overlay_scale = getattr(config.calibration, 'overlay_radius_scale', 1.15)
        
        # Calculate recommended scale
        recommended_scale = detected_radius / expected_radius if detected_radius and expected_radius > 0 else 1.0
        
        # Calculate scale that would make overlay match expected
        # overlay_radius = detected_radius * scale = expected_radius
        # So scale = expected_radius / detected_radius (inverse)
        scale_to_match_real = expected_radius / detected_radius if detected_radius and detected_radius > 0 else 1.0
        
        # Determine position in frame (9 positions)
        frame_w, frame_h = sim.camera.resolution if sim.camera else (1280, 800)
        
        # Determine horizontal position
        if ball_x < frame_w * 0.33:
            h_pos = "left"
        elif ball_x > frame_w * 0.67:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Determine vertical position
        if ball_y < frame_h * 0.33:
            v_pos = "top"
        elif ball_y > frame_h * 0.67:
            v_pos = "bottom"
        else:
            v_pos = "center"
        
        # Combine for position name
        if h_pos == "center" and v_pos == "center":
            position_name = "center"
        elif h_pos == "center":
            position_name = v_pos
        elif v_pos == "center":
            position_name = h_pos
        else:
            position_name = f"{v_pos}-{h_pos}"
        
        return JSONResponse({
            "success": True,
            "position": {
                "x_px": round(ball_x, 1),
                "y_px": round(ball_y, 1),
                "position_name": position_name,
                "normalized_x": round(ball_x / frame_w, 3),
                "normalized_y": round(ball_y / frame_h, 3)
            },
            "radius": {
                "detected_px": round(detected_radius, 2) if detected_radius else None,
                "expected_px": round(expected_radius, 2),
                "ratio_detected_to_expected": round(detected_radius / expected_radius, 3) if detected_radius and expected_radius > 0 else None
            },
            "overlay_scale": {
                "current_config_scale": current_overlay_scale,
                "recommended_scale": round(scale_to_match_real, 3),
                "scale_difference": round(scale_to_match_real - current_overlay_scale, 3)
            },
            "confidence": round(confidence, 3) if confidence else None,
            "calibration": {
                "pixels_per_meter": round(ppm, 1),
                "ball_diameter_m": BALL_DIAMETER_M
            },
            "instructions": (
                f"Current position: {position_name}. "
                f"For complete test, place ball at all 9 positions: "
                f"center, top, bottom, left, right, top-left, top-right, bottom-left, bottom-right. "
                f"Recommended overlay_radius_scale based on this position: {scale_to_match_real:.3f}"
            )
        })
        
    except Exception as e:
        logger.error(f"9-position overlay test failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# Storage for 10-shot validation report
_validation_shots = []


@app.post("/api/validation/10-shot-report")
async def ten_shot_validation_report(data: dict):
    """
    10-shot validation report endpoint.
    
    Input options:
    1. Add a shot: {"action": "add", "real_cm": 50.0}
    2. Get report: {"action": "report"}
    3. Clear data: {"action": "clear"}
    
    Returns for each shot:
    - real_cm (ground truth)
    - measured_cm (system output)
    - error_cm
    - error_percent
    - method_used (trajectory_fit, exit_velocity, v0_robust, physical_only)
    
    Plus aggregate:
    - mean_error_percent
    - max_error_percent
    - shots_within_5_percent
    """
    global _validation_shots
    
    try:
        sim = get_app_instance()
        action = data.get("action", "add")
        
        if action == "clear":
            _validation_shots = []
            return JSONResponse({
                "success": True,
                "message": "Validation data cleared",
                "shots_count": 0
            })
        
        if action == "add":
            real_cm = data.get("real_cm")
            if real_cm is None:
                return JSONResponse({
                    "success": False,
                    "error": "real_cm is required for adding a shot"
                }, status_code=400)
            
            state = sim._current_state
            if not state or not state.shot_result:
                return JSONResponse({
                    "success": False,
                    "error": "No shot recorded. Make a putt first, then add to validation."
                })
            
            result = state.shot_result
            ppm = result.pixels_per_meter if result.pixels_per_meter > 0 else sim.get_pixels_per_meter()
            
            # Get system measurements
            measured_cm = (result.total_distance_px / ppm) * 100
            physical_cm = (result.physical_distance_px / ppm) * 100
            virtual_cm = (result.virtual_distance_px / ppm) * 100
            speed_m_s = result.initial_speed_px_s / ppm
            
            # Get timing stats
            shot_timing = None
            if sim.tracker.shot_timing_stats:
                ts = sim.tracker.shot_timing_stats
                shot_timing = {
                    "effective_fps": round(ts.effective_fps, 1),
                    "dt_mean_ms": round(ts.dt_mean_ms, 2),
                    "dt_std_ms": round(ts.dt_std_ms, 2),
                    "frame_count": ts.frame_count
                }
            
            # Calculate errors
            error_cm = measured_cm - real_cm
            error_percent = (error_cm / real_cm) * 100 if real_cm > 0 else 0
            
            # Determine method used (check tracker state for distance method)
            # This is a simplified heuristic - ideally we'd track this in the tracker
            method_used = "unknown"
            if result.exited_frame:
                if hasattr(sim.tracker, '_fitted_physics') and sim.tracker._fitted_physics:
                    method_used = "trajectory_fit"
                else:
                    method_used = "exit_velocity"
            else:
                method_used = "physical_only"
            
            shot_data = {
                "shot_number": len(_validation_shots) + 1,
                "real_cm": float(round(real_cm, 1)),
                "measured_cm": float(round(measured_cm, 1)),
                "physical_cm": float(round(physical_cm, 1)),
                "virtual_cm": float(round(virtual_cm, 1)),
                "error_cm": float(round(error_cm, 2)),
                "error_percent": float(round(error_percent, 2)),
                "speed_m_s": float(round(speed_m_s, 3)),
                "direction_deg": float(round(result.initial_direction_deg, 2)),
                "exited_frame": bool(result.exited_frame),
                "method_used": method_used,
                "timing": shot_timing
            }
            
            _validation_shots.append(shot_data)
            
            logger.info(f"VALIDATION SHOT {shot_data['shot_number']}: real={real_cm:.1f}cm, "
                       f"measured={measured_cm:.1f}cm, error={error_percent:.1f}%")
            
            return JSONResponse({
                "success": True,
                "shot_added": shot_data,
                "shots_count": len(_validation_shots),
                "message": f"Shot {shot_data['shot_number']} added to validation report"
            })
        
        if action == "report":
            if len(_validation_shots) == 0:
                return JSONResponse({
                    "success": False,
                    "error": "No shots recorded. Add shots first with action='add'."
                })
            
            # Calculate aggregate statistics
            errors = [s["error_percent"] for s in _validation_shots]
            abs_errors = [abs(e) for e in errors]
            
            mean_error = sum(errors) / len(errors)
            mean_abs_error = sum(abs_errors) / len(abs_errors)
            max_abs_error = max(abs_errors)
            min_error = min(errors)
            max_error = max(errors)
            
            within_2_percent = sum(1 for e in abs_errors if e <= 2.0)
            within_5_percent = sum(1 for e in abs_errors if e <= 5.0)
            within_10_percent = sum(1 for e in abs_errors if e <= 10.0)
            
            # Check timing consistency
            timing_consistent = True
            if all(s.get("timing") for s in _validation_shots):
                fps_values = [s["timing"]["effective_fps"] for s in _validation_shots]
                fps_std = float(np.std(fps_values))
                timing_consistent = bool(fps_std < 5)  # Less than 5 fps variation
            
            report = {
                "success": True,
                "shots_count": len(_validation_shots),
                "shots": _validation_shots,
                "aggregate": {
                    "mean_error_percent": float(round(mean_error, 2)),
                    "mean_abs_error_percent": float(round(mean_abs_error, 2)),
                    "max_abs_error_percent": float(round(max_abs_error, 2)),
                    "min_error_percent": float(round(min_error, 2)),
                    "max_error_percent": float(round(max_error, 2)),
                    "within_2_percent": int(within_2_percent),
                    "within_5_percent": int(within_5_percent),
                    "within_10_percent": int(within_10_percent),
                    "timing_consistent": bool(timing_consistent)
                },
                "acceptance_criteria": {
                    "target_error_percent": 5.0,
                    "passes": bool(mean_abs_error <= 5.0 and max_abs_error <= 10.0),
                    "message": (
                        "PASSED: Mean error <= 5% and max error <= 10%"
                        if mean_abs_error <= 5.0 and max_abs_error <= 10.0
                        else f"FAILED: Mean error {mean_abs_error:.1f}% (target <=5%), "
                             f"max error {max_abs_error:.1f}% (target <=10%)"
                    )
                }
            }
            
            return JSONResponse(report)
        
        return JSONResponse({
            "success": False,
            "error": f"Unknown action: {action}. Use 'add', 'report', or 'clear'."
        }, status_code=400)
        
    except Exception as e:
        logger.error(f"10-shot validation report failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


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
    
    sim = get_app_instance()
    await sim.add_client(websocket)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                try:
                    cmd = json.loads(data)
                    if cmd.get("type") == "reset":
                        sim.reset_all()
                    elif cmd.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                # Client hasn't sent anything in 60s, send a ping to verify connection
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await sim.remove_client(websocket)
        except Exception:
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
