"""
TrackMan-style shot metrics for StrikeLab Putting Sim.

Unified data models for all shot analytics, designed for both frontend display
and future Unreal Engine API consumption.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Tuple


class ShotType(Enum):
    PUTT = "putt"
    CHIP = "chip"
    UNKNOWN = "unknown"


@dataclass
class BallMetrics:
    """Ball-specific metrics."""
    speed_m_s: float = 0.0              # Ball speed off the face
    distance_m: float = 0.0             # Total roll/carry distance
    direction_deg: float = 0.0          # Launch direction relative to target line
    launch_angle_deg: float = 0.0       # Vertical launch angle (0 for putts)
    # Putting-specific
    skid_distance_m: float = 0.0        # Distance before true roll begins
    true_roll_pct: float = 0.0          # Percentage of distance in true roll
    # Chipping-specific
    carry_distance_m: float = 0.0       # Air distance before landing
    roll_distance_m: float = 0.0        # Roll after landing
    peak_height_m: float = 0.0          # Maximum ball height
    landing_angle_deg: float = 0.0      # Angle at which ball lands
    spin_estimate_rpm: float = 0.0      # Estimated backspin from trajectory shape

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


@dataclass
class ClubMetricsReport:
    """Club-specific metrics (from ZED 2i when available)."""
    club_path_deg: float = 0.0          # Path angle at impact
    face_angle_deg: float = 0.0         # Face angle at impact
    attack_angle_deg: float = 0.0       # Vertical attack angle
    club_speed_m_s: float = 0.0         # Head speed at impact
    dynamic_loft_deg: float = 0.0       # Effective loft at impact
    impact_point_x: float = 0.0         # Strike point on face (toe-heel)
    impact_point_y: float = 0.0         # Strike point on face (high-low)
    stroke_tempo: float = 0.0           # Backswing / forward swing ratio
    backswing_time_ms: float = 0.0
    forward_swing_time_ms: float = 0.0
    backswing_length_m: float = 0.0
    forward_swing_length_m: float = 0.0
    available: bool = False             # Whether club data was captured

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


@dataclass
class ShotReport:
    """
    Complete shot report combining all available sensor data.
    This is the primary data structure for the API and frontend.
    """
    shot_id: int = 0
    timestamp_ms: float = 0.0
    shot_type: ShotType = ShotType.PUTT

    ball: BallMetrics = field(default_factory=BallMetrics)
    club: ClubMetricsReport = field(default_factory=ClubMetricsReport)

    # Trajectory data for visualization
    trajectory_2d: List[Tuple[float, float]] = field(default_factory=list)
    trajectory_3d: List[Tuple[float, float, float]] = field(default_factory=list)
    club_path_3d: List[Tuple[float, float, float]] = field(default_factory=list)

    # Game result (from existing game_logic)
    result: str = ""                    # "made", "miss_right", "miss_left", etc.
    is_made: bool = False
    distance_to_hole_m: float = 0.0

    # Metadata
    cameras_used: List[str] = field(default_factory=list)
    fast_putt_resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "timestamp_ms": round(self.timestamp_ms, 1),
            "shot_type": self.shot_type.value,
            "ball": self.ball.to_dict(),
            "club": self.club.to_dict(),
            "trajectory_2d": [(round(x, 1), round(y, 1)) for x, y in self.trajectory_2d[-50:]],
            "trajectory_3d": [(round(x, 3), round(y, 3), round(z, 3))
                              for x, y, z in self.trajectory_3d[-50:]],
            "club_path_3d": [(round(x, 3), round(y, 3), round(z, 3))
                             for x, y, z in self.club_path_3d[-30:]],
            "result": self.result,
            "is_made": self.is_made,
            "distance_to_hole_m": round(self.distance_to_hole_m, 3),
            "cameras_used": self.cameras_used,
            "fast_putt_resolved": self.fast_putt_resolved,
        }
