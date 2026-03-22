"""
ZED 2i stereo depth camera source.
Provides color + depth + point cloud at up to 60fps (HD720).
Mounted overhead at 40-50cm, angled down toward the putting surface.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Any, List

from .base import CameraSource, CameraFrame, CameraType

logger = logging.getLogger(__name__)

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    logger.warning("pyzed not installed - ZED camera will be unavailable")


class ZedSource(CameraSource):
    """
    ZED 2i depth camera using Stereolabs SDK.

    Configured for overhead mounting at 40-50cm:
    - HD720 @ 60fps (good balance of resolution and speed)
    - NEURAL depth mode for highest accuracy
    - Min depth 0.15m (close-range for overhead)
    - Ball moves right to left in the image
    """

    DEFAULT_RESOLUTION = sl.RESOLUTION.HD720 if ZED_AVAILABLE else None
    DEFAULT_FPS = 60
    DEFAULT_DEPTH_MODE = sl.DEPTH_MODE.NEURAL if ZED_AVAILABLE else None

    def __init__(
        self,
        serial_number: int = 0,
        resolution: Optional[Any] = None,
        fps: int = DEFAULT_FPS,
        depth_mode: Optional[Any] = None,
        min_depth_m: float = 0.15,
        max_depth_m: float = 2.0,
        confidence_threshold: int = 50,
        auto_exposure_gain: bool = True,
        auto_white_balance: bool = True,
        exposure: int = -1,
        gain: int = -1,
        whitebalance_temperature: int = -1,
        brightness: int = -1,
        contrast: int = -1,
        saturation: int = -1,
        sharpness: int = -1,
        gamma: int = -1,
    ):
        super().__init__(CameraType.ZED)
        self._serial = serial_number
        self._resolution = self._resolve_resolution(resolution) if resolution is not None else self.DEFAULT_RESOLUTION
        self._fps = fps
        self._depth_mode = self._resolve_depth_mode(depth_mode) if depth_mode is not None else self.DEFAULT_DEPTH_MODE
        self._min_depth = min_depth_m
        self._max_depth = max_depth_m
        self._confidence_threshold = confidence_threshold
        self._auto_exposure_gain = bool(auto_exposure_gain)
        self._auto_white_balance = bool(auto_white_balance)
        self._exposure = int(exposure)
        self._gain = int(gain)
        self._whitebalance_temperature = int(whitebalance_temperature)
        self._brightness = int(brightness)
        self._contrast = int(contrast)
        self._saturation = int(saturation)
        self._sharpness = int(sharpness)
        self._gamma = int(gamma)

        self._zed: Optional[object] = None
        self._image_mat: Optional[object] = None
        self._depth_mat: Optional[object] = None
        self._point_cloud_mat: Optional[object] = None
        self._confidence_mat: Optional[object] = None
        self._runtime_params: Optional[object] = None

    def _try_set_camera_setting(self, setting_name: str, value: int) -> bool:
        """Best-effort camera setting update; returns True when applied."""
        if not ZED_AVAILABLE or self._zed is None:
            return False
        if value is None:
            return False
        setting = getattr(sl.VIDEO_SETTINGS, setting_name, None)
        if setting is None:
            return False
        try:
            self._zed.set_camera_settings(setting, int(value))
            return True
        except Exception:
            return False

    def _apply_image_tuning(self) -> None:
        """Apply image controls without failing camera startup."""
        if not ZED_AVAILABLE or self._zed is None:
            return

        # Exposure/gain and white-balance autos provide the best adaptive image in varying light.
        self._try_set_camera_setting("AEC_AGC", 1 if self._auto_exposure_gain else 0)
        self._try_set_camera_setting("WHITEBALANCE_AUTO", 1 if self._auto_white_balance else 0)

        if not self._auto_exposure_gain:
            if self._exposure >= 0:
                self._try_set_camera_setting("EXPOSURE", self._exposure)
            if self._gain >= 0:
                self._try_set_camera_setting("GAIN", self._gain)

        if not self._auto_white_balance and self._whitebalance_temperature >= 0:
            self._try_set_camera_setting("WHITEBALANCE_TEMPERATURE", self._whitebalance_temperature)

        for setting_name, value in (
            ("BRIGHTNESS", self._brightness),
            ("CONTRAST", self._contrast),
            ("SATURATION", self._saturation),
            ("SHARPNESS", self._sharpness),
            ("GAMMA", self._gamma),
        ):
            if value >= 0:
                self._try_set_camera_setting(setting_name, value)

    @staticmethod
    def _resolve_resolution(value: Any) -> Optional[object]:
        if not ZED_AVAILABLE:
            return None
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        key = value.strip().upper()
        mapping = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        return mapping.get(key, sl.RESOLUTION.HD720)

    @staticmethod
    def _resolve_depth_mode(value: Any) -> Optional[object]:
        if not ZED_AVAILABLE:
            return None
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        key = value.strip().upper()
        mapping = {
            "NEURAL": sl.DEPTH_MODE.NEURAL,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        }
        return mapping.get(key, sl.DEPTH_MODE.NEURAL)

    @staticmethod
    def _fallback_fps_for_resolution(resolution: object) -> List[int]:
        if not ZED_AVAILABLE:
            return [60, 30, 15]
        if resolution == sl.RESOLUTION.VGA:
            return [100, 60, 30, 15]
        if resolution == sl.RESOLUTION.HD720:
            return [60, 30, 15]
        if resolution == sl.RESOLUTION.HD1080:
            return [30, 15]
        if resolution == sl.RESOLUTION.HD2K:
            return [15]
        return [60, 30, 15]

    def _candidate_init_configs(self) -> List[Tuple[object, int, object]]:
        if not ZED_AVAILABLE:
            return []
        requested_resolution = self._resolution or sl.RESOLUTION.HD720
        requested_depth_mode = self._depth_mode or sl.DEPTH_MODE.NEURAL
        requested_fps = max(1, int(self._fps))

        # Keep startup attempts tight to avoid long noisy failures.
        candidates: List[Tuple[object, int, object]] = [
            (requested_resolution, requested_fps, requested_depth_mode),
            (requested_resolution, min(requested_fps, 60), sl.DEPTH_MODE.PERFORMANCE),
            (sl.RESOLUTION.HD720, 60, sl.DEPTH_MODE.PERFORMANCE),
            (sl.RESOLUTION.HD720, 30, sl.DEPTH_MODE.PERFORMANCE),
            (sl.RESOLUTION.VGA, 100, sl.DEPTH_MODE.PERFORMANCE),
            (sl.RESOLUTION.VGA, 60, sl.DEPTH_MODE.PERFORMANCE),
        ]

        # De-duplicate while preserving order.
        deduped: List[Tuple[object, int, object]] = []
        for c in candidates:
            if c not in deduped:
                deduped.append(c)
        return deduped

    def start(self) -> bool:
        if not ZED_AVAILABLE:
            self._error = "ZED SDK (pyzed) not installed"
            logger.error(self._error)
            return False

        open_status = None
        selected = None
        for resolution, fps, depth_mode in self._candidate_init_configs():
            cam = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = resolution
            init_params.camera_fps = int(fps)
            init_params.depth_mode = depth_mode
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_minimum_distance = self._min_depth
            init_params.depth_maximum_distance = self._max_depth
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

            if self._serial > 0:
                init_params.set_from_serial_number(self._serial)

            open_status = cam.open(init_params)
            if open_status == sl.ERROR_CODE.SUCCESS:
                selected = (resolution, int(fps), depth_mode)
                self._zed = cam
                break
            # Ensure clean state before trying the next fallback profile.
            try:
                cam.close()
            except Exception:
                pass

        if selected is None:
            self._error = f"ZED open failed after fallback attempts: {open_status}"
            logger.error(self._error)
            self._zed = None
            return False

        self._resolution, self._fps, self._depth_mode = selected

        self._image_mat = sl.Mat()
        self._depth_mat = sl.Mat()
        self._point_cloud_mat = sl.Mat()
        self._confidence_mat = sl.Mat()

        self._runtime_params = sl.RuntimeParameters()
        self._runtime_params.confidence_threshold = int(self._confidence_threshold)
        self._runtime_params.texture_confidence_threshold = 100
        self._apply_image_tuning()

        info = self._zed.get_camera_information()
        res = info.camera_configuration.resolution
        actual_fps = info.camera_configuration.fps
        logger.info(
            f"ZED 2i started: {res.width}x{res.height} @ {actual_fps}fps, "
            f"depth={self._depth_mode}, serial={info.serial_number}"
        )

        self._running = True
        return True

    def stop(self) -> None:
        self._running = False
        if self._zed is not None:
            self._zed.close()
            self._zed = None
        self._image_mat = None
        self._depth_mat = None
        self._point_cloud_mat = None
        self._confidence_mat = None
        logger.info("ZedSource stopped")

    def read_frame(self) -> Optional[CameraFrame]:
        if not self._running or self._zed is None:
            return None

        err = self._zed.grab(self._runtime_params)
        if err != sl.ERROR_CODE.SUCCESS:
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                self._running = False
            return None

        # Left color image (BGR for OpenCV compatibility)
        self._zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
        color = self._image_mat.get_data().copy()
        if color.shape[2] == 4:  # BGRA -> BGR
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)

        # Depth map (float32, meters)
        self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
        depth = self._depth_mat.get_data().copy()

        # Point cloud (XYZRGBA)
        self._zed.retrieve_measure(self._point_cloud_mat, sl.MEASURE.XYZRGBA)
        pc_raw = self._point_cloud_mat.get_data().copy()
        point_cloud = pc_raw[:, :, :3]  # Keep XYZ only

        # Confidence
        self._zed.retrieve_measure(self._confidence_mat, sl.MEASURE.CONFIDENCE)
        confidence = self._confidence_mat.get_data().copy()

        timestamp_ns = self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

        self._frame_id += 1
        self._fps_counter.tick()

        return CameraFrame(
            color=color,
            timestamp_ns=timestamp_ns,
            acquisition_monotonic_ns=time.monotonic_ns(),
            frame_id=self._frame_id,
            camera_type=CameraType.ZED,
            depth=depth,
            point_cloud=point_cloud,
            confidence=confidence,
        )

    def resolution(self) -> Tuple[int, int]:
        if self._zed is not None:
            info = self._zed.get_camera_information()
            res = info.camera_configuration.resolution
            return (res.width, res.height)
        return (1280, 720)

    def get_3d_position(self, x_px: int, y_px: int) -> Optional[Tuple[float, float, float]]:
        """
        Get world (X, Y, Z) in meters for a pixel coordinate.
        Returns None if depth is invalid at that point.
        """
        if self._point_cloud_mat is None:
            return None

        err, point = self._point_cloud_mat.get_value(x_px, y_px)
        if err != sl.ERROR_CODE.SUCCESS:
            return None

        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            return None

        return (x, y, z)

    def get_depth_at(self, x_px: int, y_px: int) -> Optional[float]:
        """Get depth in meters at a pixel coordinate."""
        if self._depth_mat is None:
            return None

        err, depth_val = self._depth_mat.get_value(x_px, y_px)
        if err != sl.ERROR_CODE.SUCCESS or np.isnan(depth_val):
            return None

        return float(depth_val)
