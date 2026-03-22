"""
Intel RealSense D455 depth camera source.
Mounted at ground level, looking straight at the golf ball.
Primary purpose: launch angle detection for chipping, ball speed confirmation.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Any, List, Dict

from .base import CameraSource, CameraFrame, CameraType

logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    logger.warning("pyrealsense2 not installed - RealSense camera will be unavailable")


class RealSenseSource(CameraSource):
    """
    Intel RealSense D455 depth camera.

    Ground-level side view configuration:
    - Depth stream: 848x480 @ 60fps (good balance)
    - Color stream: 848x480 @ 60fps (aligned to depth)
    - D455 min depth: 0.6m - ball must be at least 60cm away
    - Ball moves right to left in image
    """

    DEFAULT_DEPTH_WIDTH = 848
    DEFAULT_DEPTH_HEIGHT = 480
    DEFAULT_COLOR_WIDTH = 848
    DEFAULT_COLOR_HEIGHT = 480
    DEFAULT_FPS = 60

    def __init__(
        self,
        serial_number: Optional[str] = None,
        depth_width: int = DEFAULT_DEPTH_WIDTH,
        depth_height: int = DEFAULT_DEPTH_HEIGHT,
        color_width: int = DEFAULT_COLOR_WIDTH,
        color_height: int = DEFAULT_COLOR_HEIGHT,
        fps: int = DEFAULT_FPS,
        depth_visual_preset: str = "HIGH_ACCURACY",
        depth_auto_exposure: bool = True,
        emitter_enabled: bool = True,
        laser_power: float = 330.0,
        enable_depth_post_processing: bool = True,
        color_auto_exposure: bool = True,
        color_exposure: int = 0,
        color_gain: int = 0,
        color_auto_white_balance: bool = True,
        color_white_balance: int = 0,
        color_sharpness: int = -1,
        color_contrast: int = -1,
        color_saturation: int = -1,
        color_brightness: int = -1,
    ):
        super().__init__(CameraType.REALSENSE)
        self._serial = serial_number
        self._depth_width = depth_width
        self._depth_height = depth_height
        self._color_width = color_width
        self._color_height = color_height
        self._fps = fps

        self._pipeline: Optional[object] = None
        self._align: Optional[object] = None
        self._depth_scale: float = 0.001
        self._intrinsics: Optional[object] = None
        self._color_format: Optional[Any] = None
        self._depth_fps: int = fps
        self._color_fps: int = fps
        self._depth_visual_preset = (depth_visual_preset or "HIGH_ACCURACY").strip().upper()
        self._depth_auto_exposure = bool(depth_auto_exposure)
        self._emitter_enabled = bool(emitter_enabled)
        self._laser_power = float(laser_power)
        self._enable_depth_post_processing = bool(enable_depth_post_processing)
        self._color_auto_exposure = bool(color_auto_exposure)
        self._color_exposure = int(color_exposure)
        self._color_gain = int(color_gain)
        self._color_auto_white_balance = bool(color_auto_white_balance)
        self._color_white_balance = int(color_white_balance)
        self._color_sharpness = int(color_sharpness)
        self._color_contrast = int(color_contrast)
        self._color_saturation = int(color_saturation)
        self._color_brightness = int(color_brightness)
        self._depth_filters: List[Any] = []

    def _select_with_target_fps(
        self,
        depth_profiles: List[Tuple[int, int, int, Any]],
        color_profiles: List[Tuple[int, int, int, Any]],
    ) -> Optional[Dict[str, Any]]:
        target_dw, target_dh = self._depth_width, self._depth_height
        target_cw, target_ch = self._color_width, self._color_height
        target_fps = max(1, int(self._fps))

        candidates: List[Dict[str, Any]] = []
        for dw, dh, dfps, dformat in depth_profiles:
            for cw, ch, cfps, cformat in color_profiles:
                effective_fps = min(dfps, cfps)
                candidates.append(
                    {
                        "depth": (dw, dh, dfps, dformat),
                        "color": (cw, ch, cfps, cformat),
                        "effective_fps": effective_fps,
                    }
                )

        if not candidates:
            return None

        def quality_score(c: Dict[str, Any]) -> Tuple[int, int, int]:
            dw, dh, *_ = c["depth"]
            cw, ch, *_ = c["color"]
            efps = int(c["effective_fps"])
            return (
                abs(dw - target_dw) + abs(dh - target_dh) + abs(cw - target_cw) + abs(ch - target_ch),
                abs(efps - target_fps),
                -efps,
            )

        meets_target = [c for c in candidates if int(c["effective_fps"]) >= target_fps]
        if meets_target:
            return min(meets_target, key=quality_score)

        # Fallback: prioritize highest available FPS, then quality closeness.
        return min(
            candidates,
            key=lambda c: (
                -int(c["effective_fps"]),
                abs(c["depth"][0] - target_dw) + abs(c["depth"][1] - target_dh),
                abs(c["color"][0] - target_cw) + abs(c["color"][1] - target_ch),
            ),
        )

    def _try_set_option(self, sensor: Any, option_name: str, value: float) -> bool:
        if not REALSENSE_AVAILABLE:
            return False
        option = getattr(rs.option, option_name, None)
        if option is None:
            return False
        try:
            if not sensor.supports(option):
                return False
            sensor.set_option(option, float(value))
            return True
        except Exception:
            return False

    def _resolve_visual_preset_value(self) -> Optional[float]:
        if not REALSENSE_AVAILABLE:
            return None
        preset_enum = getattr(rs, "rs400_visual_preset", None)
        if preset_enum is None:
            return None
        mapping = {
            "CUSTOM": getattr(preset_enum, "custom", None),
            "DEFAULT": getattr(preset_enum, "default", None),
            "HAND": getattr(preset_enum, "hand", None),
            "HIGH_ACCURACY": getattr(preset_enum, "high_accuracy", None),
            "HIGH_DENSITY": getattr(preset_enum, "high_density", None),
            "MEDIUM_DENSITY": getattr(preset_enum, "medium_density", None),
        }
        selected = mapping.get(self._depth_visual_preset)
        if selected is None:
            selected = mapping.get("HIGH_ACCURACY")
        if selected is None:
            return None
        try:
            return float(int(selected))
        except Exception:
            return None

    def _configure_sensors(self, profile: Any) -> None:
        if not REALSENSE_AVAILABLE:
            return
        try:
            sensors = profile.get_device().query_sensors()
        except Exception:
            return

        depth_sensor = None
        color_sensor = None
        for sensor in sensors:
            try:
                if sensor.is_depth_sensor():
                    depth_sensor = sensor
                    continue
            except Exception:
                pass
            for stream_profile in sensor.get_stream_profiles():
                if stream_profile.stream_type() == rs.stream.color:
                    color_sensor = sensor
                    break

        if depth_sensor is not None:
            preset_value = self._resolve_visual_preset_value()
            if preset_value is not None:
                self._try_set_option(depth_sensor, "visual_preset", preset_value)
            self._try_set_option(depth_sensor, "enable_auto_exposure", 1 if self._depth_auto_exposure else 0)
            self._try_set_option(depth_sensor, "emitter_enabled", 1 if self._emitter_enabled else 0)
            if self._laser_power > 0:
                self._try_set_option(depth_sensor, "laser_power", self._laser_power)

        if color_sensor is not None:
            self._try_set_option(color_sensor, "enable_auto_exposure", 1 if self._color_auto_exposure else 0)
            if not self._color_auto_exposure:
                if self._color_exposure > 0:
                    self._try_set_option(color_sensor, "exposure", self._color_exposure)
                if self._color_gain > 0:
                    self._try_set_option(color_sensor, "gain", self._color_gain)
            self._try_set_option(
                color_sensor,
                "enable_auto_white_balance",
                1 if self._color_auto_white_balance else 0,
            )
            if not self._color_auto_white_balance and self._color_white_balance > 0:
                self._try_set_option(color_sensor, "white_balance", self._color_white_balance)
            if self._color_sharpness >= 0:
                self._try_set_option(color_sensor, "sharpness", self._color_sharpness)
            if self._color_contrast >= 0:
                self._try_set_option(color_sensor, "contrast", self._color_contrast)
            if self._color_saturation >= 0:
                self._try_set_option(color_sensor, "saturation", self._color_saturation)
            if self._color_brightness >= 0:
                self._try_set_option(color_sensor, "brightness", self._color_brightness)

    def _build_depth_post_filters(self) -> List[Any]:
        if not REALSENSE_AVAILABLE or not self._enable_depth_post_processing:
            return []
        filters: List[Any] = []
        try:
            filters.append(rs.decimation_filter())
            filters.append(rs.spatial_filter())
            filters.append(rs.temporal_filter())
            filters.append(rs.hole_filling_filter())
        except Exception:
            return []
        return filters

    def _select_best_stream_pair(self) -> Optional[Dict[str, Any]]:
        """
        Select the highest-FPS compatible color+depth pair available on the device.
        Prefers requested resolutions when FPS is comparable.
        """
        if not REALSENSE_AVAILABLE:
            return None

        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                self._error = "No RealSense devices detected"
                return None

            device = None
            if self._serial:
                for dev in devices:
                    serial = dev.get_info(rs.camera_info.serial_number)
                    if serial == self._serial:
                        device = dev
                        break
                if device is None:
                    self._error = f"RealSense serial '{self._serial}' not found"
                    return None
            else:
                device = devices[0]
                try:
                    self._serial = device.get_info(rs.camera_info.serial_number)
                except Exception:
                    pass

            sensors = device.query_sensors()
            depth_sensor = None
            color_sensor = None
            for sensor in sensors:
                try:
                    if sensor.is_depth_sensor():
                        depth_sensor = sensor
                        continue
                except Exception:
                    pass
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() == rs.stream.color:
                        color_sensor = sensor
                        break
                if color_sensor is not None and depth_sensor is not None:
                    break

            if depth_sensor is None or color_sensor is None:
                self._error = "Could not find both depth and color RealSense sensors"
                return None

            depth_profiles = []
            for profile in depth_sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.depth:
                    continue
                vp = profile.as_video_stream_profile()
                if vp.format() != rs.format.z16:
                    continue
                depth_profiles.append(
                    (int(vp.width()), int(vp.height()), int(vp.fps()), vp.format())
                )

            color_profiles = []
            for profile in color_sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
                    continue
                vp = profile.as_video_stream_profile()
                if vp.format() not in (rs.format.bgr8, rs.format.rgb8):
                    continue
                color_profiles.append(
                    (int(vp.width()), int(vp.height()), int(vp.fps()), vp.format())
                )

            if not depth_profiles or not color_profiles:
                self._error = "No usable RealSense stream profiles found"
                return None

            return self._select_with_target_fps(depth_profiles, color_profiles)
        except Exception as e:
            self._error = f"RealSense profile selection failed: {e}"
            return None

    def start(self) -> bool:
        if not REALSENSE_AVAILABLE:
            self._error = "pyrealsense2 not installed"
            logger.error(self._error)
            return False

        try:
            selected = self._select_best_stream_pair()
            if selected is None:
                logger.error(self._error or "RealSense profile selection failed")
                return False

            depth_w, depth_h, depth_fps, depth_fmt = selected["depth"]
            color_w, color_h, color_fps, color_fmt = selected["color"]
            self._depth_width, self._depth_height = depth_w, depth_h
            self._color_width, self._color_height = color_w, color_h
            self._depth_fps = int(depth_fps)
            self._color_fps = int(color_fps)
            self._fps = int(selected.get("effective_fps", min(self._depth_fps, self._color_fps)))
            self._color_format = color_fmt

            self._pipeline = rs.pipeline()
            config = rs.config()

            if self._serial:
                config.enable_device(self._serial)

            config.enable_stream(
                rs.stream.depth,
                self._depth_width, self._depth_height,
                depth_fmt, self._depth_fps,
            )
            config.enable_stream(
                rs.stream.color,
                self._color_width, self._color_height,
                color_fmt, self._color_fps,
            )

            profile = self._pipeline.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()
            self._configure_sensors(profile)

            self._align = rs.align(rs.stream.color)
            self._depth_filters = self._build_depth_post_filters()

            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self._intrinsics = depth_profile.get_intrinsics()

            logger.info(
                f"RealSense D455 started: depth {self._depth_width}x{self._depth_height}, "
                f"color {self._color_width}x{self._color_height} @ depth={self._depth_fps}fps, "
                f"color={self._color_fps}fps (effective ~{self._fps}fps), "
                f"depth_scale={self._depth_scale:.6f}, post_filters={'on' if self._depth_filters else 'off'}"
            )

            self._running = True
            return True

        except Exception as e:
            self._error = f"RealSense init failed: {e}"
            logger.error(self._error)
            if self._pipeline:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None
            return False

    def stop(self) -> None:
        self._running = False
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align = None
        self._depth_filters = []
        logger.info("RealSenseSource stopped")

    def read_frame(self) -> Optional[CameraFrame]:
        if not self._running or self._pipeline is None:
            return None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except Exception:
            return None

        aligned = self._align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return None

        for depth_filter in self._depth_filters:
            try:
                depth_frame = depth_filter.process(depth_frame)
            except Exception:
                break

        color = np.asanyarray(color_frame.get_data())
        if self._color_format == rs.format.rgb8:
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Depth in meters (float32)
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self._depth_scale

        timestamp_ns = int(frames.get_timestamp() * 1_000_000)  # ms -> ns

        self._frame_id += 1
        self._fps_counter.tick()

        return CameraFrame(
            color=color,
            timestamp_ns=timestamp_ns,
            acquisition_monotonic_ns=time.monotonic_ns(),
            frame_id=self._frame_id,
            camera_type=CameraType.REALSENSE,
            depth=depth_m,
        )

    def resolution(self) -> Tuple[int, int]:
        return (self._color_width, self._color_height)

    def deproject_pixel(self, x: int, y: int, depth_m: float) -> Optional[Tuple[float, float, float]]:
        """Convert a pixel + depth to 3D point in camera space (meters)."""
        if not REALSENSE_AVAILABLE or self._intrinsics is None:
            return None
        point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [x, y], depth_m)
        return tuple(point)

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    @property
    def intrinsics(self):
        return self._intrinsics
