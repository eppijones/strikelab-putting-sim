"""
Putting-specific shot analyzer.
Computes enhanced putting metrics from fused multi-camera data.
"""

import numpy as np
import logging
from typing import Optional, List, Tuple

from .shot_metrics import ShotReport, ShotType, BallMetrics, ClubMetricsReport
from ..tracking.sensor_fusion import FusedShotReport
from ..tracking.club_tracker import ClubMetrics

logger = logging.getLogger(__name__)


class PuttingAnalyzer:
    """
    Computes putting-specific analytics from fused sensor data.

    Metrics computed:
    - Ball speed, distance, line (from Arducam - primary)
    - Club path, face angle, tempo (from ZED - when available)
    - Skid distance estimate (from trajectory analysis)
    - True roll percentage
    """

    SKID_RATIO_DEFAULT = 0.15  # Typical: ~15% of putt distance is skid

    def analyze(self, fused_report: FusedShotReport,
                trajectory: List[Tuple[float, float]],
                pixels_per_meter: float) -> ShotReport:
        """
        Build a ShotReport from fused data for a putting shot.
        """
        report = ShotReport(
            shot_type=ShotType.PUTT,
            cameras_used=list(fused_report.cameras_used),
            fast_putt_resolved=fused_report.fast_putt_resolved,
        )

        # Ball metrics from Arducam (primary)
        report.ball.speed_m_s = fused_report.ball_speed_m_s
        report.ball.distance_m = fused_report.distance_m
        report.ball.direction_deg = fused_report.direction_deg
        report.ball.launch_angle_deg = 0.0  # Putts have ~0° launch

        # Skid / true roll estimation
        # A well-struck putt skids for ~10-20% of its distance
        report.ball.skid_distance_m = fused_report.distance_m * self.SKID_RATIO_DEFAULT
        if fused_report.distance_m > 0:
            report.ball.true_roll_pct = (
                (fused_report.distance_m - report.ball.skid_distance_m)
                / fused_report.distance_m * 100
            )

        # Use launch data from RealSense if available
        if fused_report.launch:
            report.ball.launch_angle_deg = fused_report.launch.launch_angle_deg
            # Refine skid estimate: higher launch = more skid
            if fused_report.launch.launch_angle_deg > 2.0:
                skid_factor = min(0.3, 0.15 + fused_report.launch.launch_angle_deg * 0.02)
                report.ball.skid_distance_m = fused_report.distance_m * skid_factor
                if fused_report.distance_m > 0:
                    report.ball.true_roll_pct = (
                        (fused_report.distance_m - report.ball.skid_distance_m)
                        / fused_report.distance_m * 100
                    )

        # Club metrics from ZED
        if fused_report.club:
            self._apply_club_metrics(report.club, fused_report.club)

        # 2D trajectory
        report.trajectory_2d = trajectory

        # Club path for visualization
        if fused_report.club:
            # Will be populated from club tracker's stroke path
            pass

        return report

    def _apply_club_metrics(self, target: ClubMetricsReport, source: ClubMetrics) -> None:
        """Copy club metrics from tracker to report."""
        target.club_path_deg = source.club_path_deg
        target.face_angle_deg = source.face_angle_deg
        target.attack_angle_deg = source.attack_angle_deg
        target.club_speed_m_s = source.club_speed_m_s
        target.stroke_tempo = source.stroke_tempo
        target.backswing_time_ms = source.backswing_time_ms
        target.forward_swing_time_ms = source.forward_swing_time_ms
        target.backswing_length_m = source.backswing_length_m
        target.forward_swing_length_m = source.forward_swing_length_m
        target.available = True

        if source.impact_point:
            target.impact_point_x = source.impact_point[0]
            target.impact_point_y = source.impact_point[1]
