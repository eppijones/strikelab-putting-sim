"""
Chipping-specific shot analyzer.
Computes carry, landing, peak height, and spin metrics from
multi-camera fusion data.
"""

import numpy as np
import logging
from typing import Optional, List, Tuple

from .shot_metrics import ShotReport, ShotType, BallMetrics, ClubMetricsReport
from ..tracking.sensor_fusion import FusedShotReport, FusedBallState
from ..tracking.club_tracker import ClubMetrics
from ..tracking.ball_tracker_3d import BallTrajectory3D

logger = logging.getLogger(__name__)

GRAVITY = 9.81  # m/s²


class ChippingAnalyzer:
    """
    Computes chipping-specific analytics from fused sensor data.

    Key metrics:
    - Launch angle (from RealSense or ZED 3D)
    - Carry distance (from 3D trajectory or physics model)
    - Landing angle
    - Peak height
    - Spin estimate (from trajectory curvature vs ballistic prediction)
    - Attack angle (from ZED club tracker)
    """

    def analyze(
        self,
        fused_report: FusedShotReport,
        trajectory_2d: List[Tuple[float, float]],
        trajectory_3d: Optional[BallTrajectory3D],
        ball_states: List[FusedBallState],
        pixels_per_meter: float,
    ) -> ShotReport:
        """Build a ShotReport from fused data for a chipping shot."""
        report = ShotReport(
            shot_type=ShotType.CHIP,
            cameras_used=list(fused_report.cameras_used),
            fast_putt_resolved=fused_report.fast_putt_resolved,
        )

        # Ball metrics from Arducam / fused
        report.ball.speed_m_s = fused_report.ball_speed_m_s
        report.ball.distance_m = fused_report.distance_m
        report.ball.direction_deg = fused_report.direction_deg

        # Launch angle (prefer RealSense, fallback to ZED 3D)
        if fused_report.launch and fused_report.launch.confidence > 0.3:
            report.ball.launch_angle_deg = fused_report.launch.launch_angle_deg
        elif ball_states:
            airborne = [s for s in ball_states if s.is_airborne]
            if len(airborne) >= 2 and airborne[0].speed_3d_m_s > 0:
                dh = airborne[1].height_above_surface - airborne[0].height_above_surface
                dx = airborne[0].speed_3d_m_s * 0.016  # ~1 frame dt
                if dx > 0:
                    report.ball.launch_angle_deg = np.degrees(np.arctan2(dh, dx))

        # Peak height
        report.ball.peak_height_m = fused_report.peak_height_m
        if ball_states:
            max_h = max((s.height_above_surface for s in ball_states), default=0)
            if max_h > report.ball.peak_height_m:
                report.ball.peak_height_m = max_h

        # Carry and roll distance estimation
        carry, roll = self._estimate_carry_roll(
            report.ball.speed_m_s,
            report.ball.launch_angle_deg,
            report.ball.distance_m,
            report.ball.peak_height_m,
        )
        report.ball.carry_distance_m = carry
        report.ball.roll_distance_m = roll

        # Landing angle estimation
        report.ball.landing_angle_deg = self._estimate_landing_angle(
            report.ball.launch_angle_deg,
            report.ball.peak_height_m,
            carry,
        )

        # Spin estimate from trajectory shape
        report.ball.spin_estimate_rpm = self._estimate_spin(
            report.ball.speed_m_s,
            report.ball.launch_angle_deg,
            carry,
            report.ball.peak_height_m,
        )

        # Club metrics
        if fused_report.club:
            self._apply_club_metrics(report.club, fused_report.club)

        # Trajectories
        report.trajectory_2d = trajectory_2d
        if trajectory_3d:
            report.trajectory_3d = [
                (p.x, p.y, p.z) for p in trajectory_3d.positions
            ]

        return report

    def _estimate_carry_roll(
        self, speed: float, launch_angle: float, total_distance: float, peak_height: float,
    ) -> Tuple[float, float]:
        """Estimate carry and roll distance from launch parameters."""
        if launch_angle < 3.0:
            # Essentially a putt - all roll
            return 0.0, total_distance

        # Physics-based carry estimate
        angle_rad = np.radians(launch_angle)
        v_horiz = speed * np.cos(angle_rad)
        v_vert = speed * np.sin(angle_rad)

        if v_vert > 0:
            # Time of flight (no spin, no drag - simplified)
            t_flight = 2 * v_vert / GRAVITY
            carry = v_horiz * t_flight

            # Use peak height to refine if available
            if peak_height > 0.01:
                t_up = np.sqrt(2 * peak_height / GRAVITY)
                t_flight_from_height = 2 * t_up
                carry_from_height = v_horiz * t_flight_from_height
                carry = (carry + carry_from_height) / 2  # Average both estimates
        else:
            carry = 0.0

        carry = min(carry, total_distance)
        roll = max(0.0, total_distance - carry)

        return carry, roll

    def _estimate_landing_angle(
        self, launch_angle: float, peak_height: float, carry: float,
    ) -> float:
        """Estimate landing angle from trajectory parameters."""
        if carry < 0.01 or peak_height < 0.005:
            return 0.0

        # For a parabolic trajectory: landing angle ≈ launch angle (no spin/drag)
        # With backspin, landing angle is steeper
        landing = launch_angle * 1.1  # Slightly steeper due to drag
        return min(landing, 80.0)

    def _estimate_spin(
        self, speed: float, launch_angle: float, carry: float, peak_height: float,
    ) -> float:
        """
        Rough spin estimate from trajectory shape.

        Compare actual carry to ballistic (no-spin) carry:
        - Less carry than ballistic = backspin (ball climbs and drops steeply)
        - More carry = topspin or low spin
        """
        if launch_angle < 3.0 or speed < 0.5:
            return 0.0

        angle_rad = np.radians(launch_angle)
        ballistic_carry = (speed ** 2 * np.sin(2 * angle_rad)) / GRAVITY

        if ballistic_carry < 0.01:
            return 0.0

        ratio = carry / ballistic_carry
        # Typical chip: ratio 0.5-0.8 = backspin 3000-6000 RPM
        if ratio < 1.0:
            spin_rpm = (1.0 - ratio) * 8000  # Rough mapping
        else:
            spin_rpm = 500  # Low spin

        return max(0, min(spin_rpm, 10000))

    def _apply_club_metrics(self, target: ClubMetricsReport, source: ClubMetrics) -> None:
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
