"""
Lightweight smoke tests for tracking reliability hardening.

Covers:
  1. reset_all clears all shot artifacts
  2. calibration update preserves tuning fields
  3. cumulative path distance vs chord distance
  4. fast_putt_estimated flag propagation

Run:
    pytest tests/test_hardening.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from backend.tracker import BallTracker, ShotState
from backend.config import ConfigManager, CalibrationData
from backend.detector import BallDetector


# ---------------------------------------------------------------------------
# 1. Cumulative path distance helper
# ---------------------------------------------------------------------------

class TestCumulativePathDistance:
    """Verify _cumulative_path_distance returns polyline length, not chord."""

    def test_straight_line_matches_chord(self):
        points = [(0, 0), (3, 4)]
        dist = BallTracker._cumulative_path_distance(points)
        assert abs(dist - 5.0) < 1e-6

    def test_polyline_exceeds_chord(self):
        # Right-angle path: (0,0)->(3,0)->(3,4)  => 3 + 4 = 7
        points = [(0, 0), (3, 0), (3, 4)]
        dist = BallTracker._cumulative_path_distance(points)
        assert abs(dist - 7.0) < 1e-6
        chord = np.sqrt(3**2 + 4**2)
        assert dist > chord

    def test_empty_and_single_point(self):
        assert BallTracker._cumulative_path_distance([]) == 0.0
        assert BallTracker._cumulative_path_distance([(5, 5)]) == 0.0

    def test_many_segments(self):
        # 10 unit steps along x-axis => total 10
        points = [(i, 0) for i in range(11)]
        dist = BallTracker._cumulative_path_distance(points)
        assert abs(dist - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Calibration preservation
# ---------------------------------------------------------------------------

class TestCalibrationPreservation:
    """update_calibration must preserve user-tuned fields."""

    def test_tuning_fields_survive_update(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{}")

        mgr = ConfigManager(config_path=cfg_path)
        mgr.load()

        # Set tuning fields to non-default values
        mgr.config.calibration.distance_scale_factor = 1.12
        mgr.config.calibration.manual_pixels_per_meter = 1135.0
        mgr.config.calibration.virtual_deceleration_m_s2 = 0.48
        mgr.config.calibration.overlay_radius_scale = 1.20

        # Simulate a homography calibration save
        mgr.update_calibration(
            homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pixels_per_meter=1500.0,
            origin_px=(100, 100),
            forward_direction_deg=5.0,
        )

        cal = mgr.config.calibration
        assert cal.distance_scale_factor == 1.12
        assert cal.manual_pixels_per_meter == 1135.0
        assert cal.virtual_deceleration_m_s2 == 0.48
        assert cal.overlay_radius_scale == 1.20
        # Verify the homography fields actually updated
        assert cal.pixels_per_meter == 1500.0
        assert cal.forward_direction_deg == 5.0


# ---------------------------------------------------------------------------
# 3. Tracker reset clears fast_putt_estimated
# ---------------------------------------------------------------------------

class TestTrackerResetState:
    """Verify tracker.reset() clears shot-level flags."""

    def _make_tracker(self):
        det = BallDetector()
        return BallTracker(detector=det)

    def test_reset_clears_fast_putt_estimated(self):
        t = self._make_tracker()
        t._fast_putt_estimated = True
        t.reset()
        assert t._fast_putt_estimated is False

    def test_reset_clears_shot_result(self):
        t = self._make_tracker()
        t._shot_result = "dummy"
        t.reset()
        assert t._shot_result is None

    def test_reset_returns_to_armed(self):
        t = self._make_tracker()
        t._state = ShotState.TRACKING
        t.reset()
        assert t._state == ShotState.ARMED


# ---------------------------------------------------------------------------
# 4. ShotResult carries fast_putt_estimated
# ---------------------------------------------------------------------------

class TestShotResultFlag:
    """ShotResult.fast_putt_estimated must default False and be settable."""

    def test_default_false(self):
        from backend.tracker import ShotResult
        sr = ShotResult(
            initial_speed_px_s=0,
            initial_direction_deg=0,
            frames_to_tracking=0,
            frames_to_speed=0,
            trajectory=[],
            duration_ms=0,
        )
        assert sr.fast_putt_estimated is False

    def test_can_set_true(self):
        from backend.tracker import ShotResult
        sr = ShotResult(
            initial_speed_px_s=0,
            initial_direction_deg=0,
            frames_to_tracking=0,
            frames_to_speed=0,
            trajectory=[],
            duration_ms=0,
            fast_putt_estimated=True,
        )
        assert sr.fast_putt_estimated is True
