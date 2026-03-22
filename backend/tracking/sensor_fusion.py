"""
Multi-camera sensor fusion for StrikeLab Putting Sim.

Time-synchronizes and merges data from all cameras into a unified shot timeline:
- Arducam 2D ball tracking (primary speed/line)
- ZED 2i 3D ball + club tracking
- RealSense launch angle detection

Produces a fused shot record that combines the best data from each source.
"""

import numpy as np
import logging
import time
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from .ball_tracker_3d import Ball3DState, BallFlightPhase
from .club_tracker import ClubMetrics, StrokePhase
from .launch_detector import LaunchData, ShotType
from .fast_putt_resolver import FastPuttResolver, ResolvedSpeed
from .shot_metrics_fusion import fuse_speed_and_direction, classify_fusion_quality

logger = logging.getLogger(__name__)


@dataclass
class SensorFusionPolicy:
    """Runtime fusion policy (from config)."""
    enable_speed_fusion: bool = True
    weight_arducam: float = 0.5
    weight_zed: float = 0.35
    weight_realsense: float = 0.15
    speed_inconsistency_threshold_m_s: float = 0.45
    direction_inconsistency_threshold_deg: float = 20.0
    sync_tolerance_ms: int = 20
    enable_direction_fusion: bool = False
    allow_realsense_speed_fusion: bool = False
    sensor_direction_alignment_valid: bool = False
    require_all_cameras_for_full_quality: bool = False


@dataclass
class FusedBallState:
    """Fused ball state from all cameras."""
    # 2D from Arducam (primary)
    x_px: float = 0.0
    y_px: float = 0.0
    speed_2d_m_s: float = 0.0
    direction_deg: float = 0.0

    # 3D from ZED (when available)
    x_3d: Optional[float] = None
    y_3d: Optional[float] = None
    z_3d: Optional[float] = None
    height_above_surface: float = 0.0
    speed_3d_m_s: float = 0.0

    # Flight info
    flight_phase: str = "grounded"
    is_airborne: bool = False

    def to_dict(self) -> dict:
        d = {
            "x_px": round(self.x_px, 1),
            "y_px": round(self.y_px, 1),
            "speed_2d_m_s": round(self.speed_2d_m_s, 3),
            "direction_deg": round(self.direction_deg, 1),
            "flight_phase": self.flight_phase,
            "is_airborne": self.is_airborne,
            "height_above_surface_m": round(self.height_above_surface, 4),
        }
        if self.x_3d is not None:
            d["position_3d"] = {
                "x": round(self.x_3d, 4),
                "y": round(self.y_3d or 0, 4),
                "z": round(self.z_3d or 0, 4),
            }
            d["speed_3d_m_s"] = round(self.speed_3d_m_s, 3)
        return d


@dataclass
class FusedShotReport:
    """Complete fused shot report combining all camera data."""
    # Core metrics (always available from Arducam)
    ball_speed_m_s: float = 0.0
    distance_m: float = 0.0
    direction_deg: float = 0.0

    # Club metrics (from ZED when available)
    club: Optional[ClubMetrics] = None

    # Launch data (from RealSense when available)
    launch: Optional[LaunchData] = None

    # Shot classification
    shot_type: str = "putt"

    # 3D data
    peak_height_m: float = 0.0
    carry_distance_m: float = 0.0
    total_distance_m: float = 0.0

    # Fusion quality
    cameras_used: List[str] = field(default_factory=list)
    fast_putt_resolved: bool = False
    fast_putt_estimated: bool = False
    fusion_quality: str = "minimal"  # full | partial | minimal
    missing_streams: List[str] = field(default_factory=list)
    sensor_inconsistency: bool = False
    inconsistency_detail: Optional[str] = None
    excluded_from_official_stats: bool = False
    speed_fusion_applied: bool = False
    # Primary metrics are always from Arducam pipeline unless speed_fusion_applied overwrites
    arducam_speed_m_s: float = 0.0
    arducam_direction_deg: float = 0.0
    primary_source: str = "arducam"
    fusion_accepted: bool = False
    fusion_rejected_reason: Optional[str] = None
    sync_deltas_ms: Dict[str, float] = field(default_factory=dict)
    sensor_confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "ball_speed_m_s": round(self.ball_speed_m_s, 3),
            "distance_m": round(self.distance_m, 3),
            "direction_deg": round(self.direction_deg, 1),
            "shot_type": self.shot_type,
            "peak_height_m": round(self.peak_height_m, 3),
            "carry_distance_m": round(self.carry_distance_m, 3),
            "total_distance_m": round(self.total_distance_m, 3),
            "cameras_used": self.cameras_used,
            "fast_putt_resolved": self.fast_putt_resolved,
            "fast_putt_estimated": self.fast_putt_estimated,
            "fusion_quality": self.fusion_quality,
            "missing_streams": list(self.missing_streams),
            "sensor_inconsistency": self.sensor_inconsistency,
            "inconsistency_detail": self.inconsistency_detail,
            "excluded_from_official_stats": self.excluded_from_official_stats,
            "speed_fusion_applied": self.speed_fusion_applied,
            "arducam_speed_m_s": round(self.arducam_speed_m_s, 3),
            "arducam_direction_deg": round(self.arducam_direction_deg, 1),
            "primary_source": self.primary_source,
            "fusion_accepted": self.fusion_accepted,
            "fusion_rejected_reason": self.fusion_rejected_reason,
            "sync_deltas_ms": {k: round(v, 2) for k, v in self.sync_deltas_ms.items()},
            "sensor_confidence": {k: round(v, 3) for k, v in self.sensor_confidence.items()},
        }
        if self.club:
            d["club"] = self.club.to_dict()
        if self.launch:
            d["launch"] = self.launch.to_dict()
        return d


class SensorFusion:
    """
    Fuses data from Arducam, ZED, and RealSense into unified shot data.

    Runs continuously, collecting data from each camera's tracker, and
    produces fused reports when shots complete.
    """

    def __init__(self, pixels_per_meter: float = 1150.0):
        self._ppm = pixels_per_meter
        self._fast_putt_resolver = FastPuttResolver(pixels_per_meter)
        self._policy = SensorFusionPolicy()

        # Latest state from each camera's tracker
        self._ball_3d: Optional[Ball3DState] = None
        self._club_metrics: Optional[ClubMetrics] = None
        self._launch_data: Optional[LaunchData] = None

        # Accumulated data during a shot
        self._shot_active = False
        self._shot_ball_states: List[FusedBallState] = []
        self._shot_start_ns: int = 0

    def set_policy(self, policy: SensorFusionPolicy) -> None:
        """Update fusion weights / flags from config (call after load)."""
        self._policy = policy

    def set_calibration(self, pixels_per_meter: float, forward_direction_deg: float = 0.0) -> None:
        self._ppm = pixels_per_meter
        self._fast_putt_resolver.set_calibration(pixels_per_meter, forward_direction_deg)

    # --- Feed methods (called by each camera's processing) ---

    def feed_ball_3d(self, state: Ball3DState) -> None:
        """Feed ball 3D state from ZED tracker."""
        self._ball_3d = state
        if state.speed_m_s > 0.1:
            direction = 0.0
            if state.velocity_3d:
                direction = np.degrees(np.arctan2(state.velocity_3d[2], state.velocity_3d[0]))
            self._fast_putt_resolver.feed_zed_speed(
                state.speed_m_s, direction, state.position.timestamp_ns if state.position else 0
            )

    def feed_club_metrics(self, metrics: ClubMetrics) -> None:
        """Feed club metrics from ZED club tracker."""
        self._club_metrics = metrics

    def feed_launch_data(self, data: LaunchData) -> None:
        """Feed launch data from RealSense."""
        self._launch_data = data
        if data.ball_speed_m_s > 0.1:
            ts = data.timestamp_ns if getattr(data, "timestamp_ns", 0) else int(time.time() * 1e9)
            self._fast_putt_resolver.feed_realsense_speed(
                data.ball_speed_m_s, 0.0, ts
            )

    # --- Shot lifecycle ---

    def on_shot_start(self, timestamp_ns: Optional[int] = None) -> None:
        """Called when Arducam tracker transitions to TRACKING."""
        self._shot_active = True
        self._shot_ball_states.clear()
        self._shot_start_ns = int(timestamp_ns or time.time_ns())

    def on_shot_end(self) -> None:
        """Called when shot completes."""
        self._shot_active = False

    # --- Fast putt resolution ---

    def resolve_fast_putt(self, event_timestamp_ns: int) -> ResolvedSpeed:
        """Resolve speed for a fast putt that the Arducam missed."""
        return self._fast_putt_resolver.resolve(event_timestamp_ns)

    # --- Fused state ---

    def get_fused_ball_state(
        self,
        arducam_x: float, arducam_y: float,
        arducam_speed_m_s: float, arducam_direction: float,
    ) -> FusedBallState:
        """Combine Arducam 2D data with ZED 3D data."""
        state = FusedBallState(
            x_px=arducam_x,
            y_px=arducam_y,
            speed_2d_m_s=arducam_speed_m_s,
            direction_deg=arducam_direction,
        )

        if self._ball_3d and self._ball_3d.position:
            state.x_3d = self._ball_3d.position.x
            state.y_3d = self._ball_3d.position.y
            state.z_3d = self._ball_3d.position.z
            state.height_above_surface = self._ball_3d.height_above_surface
            state.speed_3d_m_s = self._ball_3d.speed_m_s
            state.flight_phase = self._ball_3d.flight_phase.value
            state.is_airborne = self._ball_3d.flight_phase in (
                BallFlightPhase.LAUNCHING,
                BallFlightPhase.AIRBORNE,
            )

        if self._shot_active:
            self._shot_ball_states.append(state)

        return state

    def build_shot_report(
        self,
        ball_speed_m_s: float,
        distance_m: float,
        direction_deg: float,
        shot_timestamp_ns: Optional[int] = None,
        require_all_cameras_for_official: bool = False,
    ) -> FusedShotReport:
        """Build the final fused shot report after a shot completes."""
        ard_s, ard_d = ball_speed_m_s, direction_deg

        zed_speed: Optional[float] = None
        zed_dir: Optional[float] = None
        zed_ts_ns: Optional[int] = None
        if self._ball_3d and self._ball_3d.speed_m_s > 0.05:
            zed_speed = float(self._ball_3d.speed_m_s)
            if self._ball_3d.velocity_3d:
                zed_dir = float(np.degrees(np.arctan2(
                    self._ball_3d.velocity_3d[2],
                    self._ball_3d.velocity_3d[0],
                )))
            if self._ball_3d.position:
                zed_ts_ns = int(self._ball_3d.position.timestamp_ns)

        rs_speed: Optional[float] = None
        rs_ts_ns: Optional[int] = None
        if self._launch_data and self._launch_data.confidence > 0.3:
            rs_speed = float(self._launch_data.ball_speed_m_s)
            rs_ts_ns = int(getattr(self._launch_data, "timestamp_ns", 0) or 0)

        pol = self._policy
        fused_s, fused_d = ard_s, ard_d
        fusion_applied = False
        inc_flag = False
        inc_detail: Optional[str] = None
        fusion_rejected_reason: Optional[str] = None

        event_ts_ns = int(shot_timestamp_ns or self._shot_start_ns or int(time.time() * 1e9))
        tol_ns = max(1, int(pol.sync_tolerance_ms)) * 1_000_000
        sync_deltas_ms: Dict[str, float] = {}
        if zed_ts_ns:
            sync_deltas_ms["zed"] = abs(zed_ts_ns - event_ts_ns) / 1e6
        if rs_ts_ns:
            sync_deltas_ms["realsense"] = abs(rs_ts_ns - event_ts_ns) / 1e6

        zed_synced = bool(zed_speed is not None and zed_ts_ns and abs(zed_ts_ns - event_ts_ns) <= tol_ns)
        rs_synced = bool(rs_speed is not None and rs_ts_ns and abs(rs_ts_ns - event_ts_ns) <= tol_ns)
        if zed_speed is not None and not zed_synced:
            logger.info("ZED fusion rejected: out of sync (delta=%.1fms > %dms)", sync_deltas_ms.get("zed", -1.0), pol.sync_tolerance_ms)
        if rs_speed is not None and not rs_synced:
            logger.info("RealSense fusion rejected: out of sync (delta=%.1fms > %dms)", sync_deltas_ms.get("realsense", -1.0), pol.sync_tolerance_ms)

        sensor_confidence = {
            "arducam": 0.95,
            "zed": 0.0,
            "realsense": 0.0,
        }
        if zed_synced and zed_speed is not None:
            dt_norm = min(1.0, abs(zed_ts_ns - event_ts_ns) / tol_ns) if zed_ts_ns else 1.0
            sensor_confidence["zed"] = max(0.05, 0.80 * (1.0 - dt_norm))
        if rs_synced and rs_speed is not None:
            dt_norm = min(1.0, abs(rs_ts_ns - event_ts_ns) / tol_ns) if rs_ts_ns else 1.0
            rs_quality = float(getattr(self._launch_data, "confidence", 0.3) if self._launch_data else 0.3)
            sensor_confidence["realsense"] = max(0.05, 0.65 * rs_quality * (1.0 - dt_norm))

        use_zed = zed_synced and zed_speed is not None
        use_rs = rs_synced and rs_speed is not None and pol.allow_realsense_speed_fusion

        if use_zed and zed_dir is not None:
            zed_dir_diff = abs(((zed_dir - ard_d + 180.0) % 360.0) - 180.0)
            if zed_dir_diff > pol.direction_inconsistency_threshold_deg:
                inc_flag = True
                inc_detail = (
                    f"dir_arducam_vs_zed: {ard_d:.1f} vs {zed_dir:.1f} deg "
                    f"(delta={zed_dir_diff:.1f} deg)"
                )
                logger.info("Sensor inconsistency flagged: %s", inc_detail)
                # Protect Arducam solution when direction frames are not calibrated/aligned.
                if not pol.sensor_direction_alignment_valid:
                    use_zed = False
                    fusion_rejected_reason = "direction_disagreement_unaligned_frames"

        if pol.enable_speed_fusion and (use_zed or use_rs):
            w_a = pol.weight_arducam * sensor_confidence["arducam"]
            w_z = pol.weight_zed * sensor_confidence["zed"] if use_zed else 0.0
            w_r = pol.weight_realsense * sensor_confidence["realsense"] if use_rs else 0.0
            fusion_out = fuse_speed_and_direction(
                ard_s,
                ard_d,
                zed_speed if use_zed else None,
                zed_dir if (use_zed and pol.enable_direction_fusion and pol.sensor_direction_alignment_valid) else None,
                rs_speed if use_rs else None,
                w_a,
                w_z,
                w_r,
                pol.speed_inconsistency_threshold_m_s,
            )
            fused_s = fusion_out.speed_m_s
            # Keep Arducam direction unless explicit directional fusion is enabled.
            fused_d = fusion_out.direction_deg if (pol.enable_direction_fusion and pol.sensor_direction_alignment_valid) else ard_d
            fusion_applied = True
            inc_flag = fusion_out.sensor_inconsistency
            inc_detail = fusion_out.inconsistency_detail or inc_detail
        elif pol.enable_speed_fusion and (zed_speed is not None or rs_speed is not None):
            fusion_rejected_reason = fusion_rejected_reason or "no_secondary_sensor_in_sync"

        report = FusedShotReport(
            ball_speed_m_s=fused_s,
            distance_m=distance_m,
            direction_deg=fused_d,
            cameras_used=["arducam"],
            arducam_speed_m_s=ard_s,
            arducam_direction_deg=ard_d,
            sensor_inconsistency=inc_flag,
            inconsistency_detail=inc_detail,
            speed_fusion_applied=fusion_applied,
            primary_source="arducam",
            fusion_accepted=fusion_applied,
            fusion_rejected_reason=fusion_rejected_reason,
            sync_deltas_ms=sync_deltas_ms,
            sensor_confidence=sensor_confidence,
        )

        has_zed_ball = False
        if self._shot_ball_states:
            has_zed_ball = any(s.speed_3d_m_s > 0.05 for s in self._shot_ball_states)
        if self._ball_3d and self._ball_3d.speed_m_s > 0.05:
            has_zed_ball = True

        if has_zed_ball and "zed" not in report.cameras_used:
            report.cameras_used.append("zed")

        has_zed_club = (
            self._club_metrics is not None
            and self._club_metrics.stroke_phase == StrokePhase.COMPLETE
        )
        if has_zed_club:
            report.club = self._club_metrics
            if "zed" not in report.cameras_used:
                report.cameras_used.append("zed")

        has_rs_launch = self._launch_data is not None and self._launch_data.confidence > 0.3
        if has_rs_launch and self._launch_data is not None:
            report.launch = self._launch_data
            report.peak_height_m = self._launch_data.peak_height_m
            if "realsense" not in report.cameras_used:
                report.cameras_used.append("realsense")
            if self._launch_data.shot_type == ShotType.CHIP:
                report.shot_type = "chip"
            else:
                report.shot_type = "putt"

        if self._shot_ball_states:
            max_height = max(s.height_above_surface for s in self._shot_ball_states)
            if max_height > report.peak_height_m:
                report.peak_height_m = max_height

        report.total_distance_m = distance_m

        qual, missing = classify_fusion_quality(
            True,
            has_zed_ball,
            bool(has_zed_club),
            has_rs_launch,
        )
        report.fusion_quality = qual
        report.missing_streams = missing
        if require_all_cameras_for_official and qual != "full":
            report.excluded_from_official_stats = True

        # Reset shot-level data
        self._club_metrics = None
        self._launch_data = None
        self._ball_3d = None
        self._shot_ball_states.clear()

        return report

    def reset(self) -> None:
        """Clear all accumulated shot data and intermediate caches."""
        self._ball_3d = None
        self._club_metrics = None
        self._launch_data = None
        self._shot_active = False
        self._shot_ball_states.clear()

    @property
    def shot_ball_states(self) -> List[FusedBallState]:
        return list(self._shot_ball_states)

    @property
    def fast_putt_resolver(self) -> FastPuttResolver:
        return self._fast_putt_resolver
