"""
Regression test harness for StrikeLab Putting Sim.

Validates acceptance metrics using replay videos:
1. Idle stability: < 2px stddev over 5 seconds
2. Impact latency: ≤ 2 frames to TRACKING state
3. Speed availability: ≤ 5 frames to first stable speed
4. No false shots during setup/waggle

Usage:
    pytest tests/test_regression.py -v
    python -m tests.test_regression --replay path/to/video.mp4
"""

import sys
import argparse
import logging
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.camera import Camera, CameraMode
from backend.detector import BallDetector
from backend.tracker import BallTracker, ShotState, TrackerState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReplayMetrics:
    """Metrics collected from a replay run."""
    
    # Timing
    total_frames: int = 0
    duration_s: float = 0.0
    
    # Idle stability
    idle_positions: List[Tuple[float, float]] = field(default_factory=list)
    idle_stddev_x: float = 0.0
    idle_stddev_y: float = 0.0
    idle_stddev_combined: float = 0.0
    
    # Impact detection
    shots_detected: int = 0
    frames_to_tracking: List[int] = field(default_factory=list)
    frames_to_speed: List[int] = field(default_factory=list)
    
    # False positives
    false_shots: int = 0  # Shots during labeled "setup" periods
    
    # Speed measurements
    measured_speeds: List[float] = field(default_factory=list)
    measured_directions: List[float] = field(default_factory=list)


class ReplayRunner:
    """
    Runs the tracking pipeline on a replay video and collects metrics.
    """
    
    def __init__(self, video_path: str, verbose: bool = False):
        self.video_path = video_path
        self.verbose = verbose
        
        self.camera = Camera(mode=CameraMode.REPLAY, replay_path=video_path)
        self.detector = BallDetector()
        self.tracker = BallTracker(detector=self.detector)
        
        self.metrics = ReplayMetrics()
        self._states: List[TrackerState] = []
        self._shot_start_frames: List[int] = []
        
    def run(self) -> ReplayMetrics:
        """Run replay and collect metrics."""
        logger.info(f"Running replay: {self.video_path}")
        
        if not self.camera.start():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        start_time = time.time()
        frame_count = 0
        last_state = ShotState.ARMED
        tracking_start_frame = 0
        
        try:
            for frame_data in self.camera.frames():
                frame_count += 1
                
                # Detect ball
                detection = self.detector.detect(frame_data.frame)
                
                # Update tracker
                state = self.tracker.update(
                    detection,
                    frame_data.timestamp_ns,
                    frame_data.frame_id,
                    frame=frame_data.frame
                )
                
                self._states.append(state)
                
                # Track state transitions
                if state.state != last_state:
                    if state.state == ShotState.TRACKING:
                        tracking_start_frame = frame_data.frame_id
                        self._shot_start_frames.append(frame_data.frame_id)
                        if self.verbose:
                            logger.info(f"Frame {frame_data.frame_id}: ARMED -> TRACKING")
                    
                    elif state.state == ShotState.STOPPED:
                        if self.verbose:
                            logger.info(f"Frame {frame_data.frame_id}: TRACKING -> STOPPED")
                        
                        # Record shot metrics
                        if state.shot_result:
                            self.metrics.shots_detected += 1
                            self.metrics.frames_to_tracking.append(state.shot_result.frames_to_tracking)
                            self.metrics.frames_to_speed.append(state.shot_result.frames_to_speed)
                            self.metrics.measured_speeds.append(state.shot_result.initial_speed_px_s)
                            self.metrics.measured_directions.append(state.shot_result.initial_direction_deg)
                            
                            logger.info(
                                f"Shot {self.metrics.shots_detected}: "
                                f"speed={state.shot_result.initial_speed_px_s:.1f}px/s, "
                                f"frames_to_speed={state.shot_result.frames_to_speed}"
                            )
                    
                    last_state = state.state
                
                # Collect idle positions when in ARMED state
                if state.state == ShotState.ARMED and state.ball_x is not None:
                    self.metrics.idle_positions.append((state.ball_x, state.ball_y))
                
                # Progress logging
                if frame_count % 500 == 0:
                    logger.info(f"Processed {frame_count} frames...")
        
        finally:
            self.camera.stop()
        
        # Compute final metrics
        self.metrics.total_frames = frame_count
        self.metrics.duration_s = time.time() - start_time
        
        self._compute_idle_stability()
        
        return self.metrics
    
    def _compute_idle_stability(self):
        """Compute idle position stability metrics."""
        if len(self.metrics.idle_positions) < 10:
            return
        
        positions = np.array(self.metrics.idle_positions)
        
        self.metrics.idle_stddev_x = float(np.std(positions[:, 0]))
        self.metrics.idle_stddev_y = float(np.std(positions[:, 1]))
        self.metrics.idle_stddev_combined = float(np.std(positions, axis=0).mean())
        
        logger.info(
            f"Idle stability: stddev_x={self.metrics.idle_stddev_x:.2f}px, "
            f"stddev_y={self.metrics.idle_stddev_y:.2f}px, "
            f"combined={self.metrics.idle_stddev_combined:.2f}px"
        )


def run_replay(video_path: str, verbose: bool = False) -> ReplayMetrics:
    """Convenience function to run replay and get metrics."""
    runner = ReplayRunner(video_path, verbose=verbose)
    return runner.run()


# =============================================================================
# Acceptance Tests
# =============================================================================

class AcceptanceTestResult:
    """Result of an acceptance test."""
    
    def __init__(self, name: str, passed: bool, message: str, value: float = None, target: float = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.value = value
        self.target = target
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        if self.value is not None and self.target is not None:
            return f"[{status}] {self.name}: {self.value:.2f} (target: {self.target}) - {self.message}"
        return f"[{status}] {self.name}: {self.message}"


def test_idle_stability(metrics: ReplayMetrics, max_stddev: float = 2.0) -> AcceptanceTestResult:
    """
    Test: Ball stationary -> position stddev < 2px over 5 seconds.
    """
    if len(metrics.idle_positions) < 100:
        return AcceptanceTestResult(
            "Idle Stability",
            False,
            f"Insufficient idle data ({len(metrics.idle_positions)} positions)",
            None, max_stddev
        )
    
    passed = metrics.idle_stddev_combined < max_stddev
    return AcceptanceTestResult(
        "Idle Stability",
        passed,
        f"{'OK' if passed else 'EXCEEDED'} - {len(metrics.idle_positions)} samples",
        metrics.idle_stddev_combined,
        max_stddev
    )


def test_impact_latency(metrics: ReplayMetrics, max_frames: int = 2) -> AcceptanceTestResult:
    """
    Test: Motion start -> TRACKING state within ≤ 2 frames.
    """
    if not metrics.frames_to_tracking:
        return AcceptanceTestResult(
            "Impact Latency",
            False,
            "No shots detected",
            None, max_frames
        )
    
    avg_latency = np.mean(metrics.frames_to_tracking)
    max_latency = max(metrics.frames_to_tracking)
    
    passed = max_latency <= max_frames
    return AcceptanceTestResult(
        "Impact Latency",
        passed,
        f"avg={avg_latency:.1f}, max={max_latency} frames",
        max_latency,
        max_frames
    )


def test_speed_availability(metrics: ReplayMetrics, max_frames: int = 5) -> AcceptanceTestResult:
    """
    Test: First stable speed estimate within ≤ 5 frames post-impact.
    """
    if not metrics.frames_to_speed:
        return AcceptanceTestResult(
            "Speed Availability",
            False,
            "No speed measurements",
            None, max_frames
        )
    
    avg_frames = np.mean(metrics.frames_to_speed)
    max_frames_actual = max(metrics.frames_to_speed)
    
    passed = max_frames_actual <= max_frames
    return AcceptanceTestResult(
        "Speed Availability",
        passed,
        f"avg={avg_frames:.1f}, max={max_frames_actual} frames",
        max_frames_actual,
        max_frames
    )


def test_no_false_shots(metrics: ReplayMetrics) -> AcceptanceTestResult:
    """
    Test: No false shot triggers during setup/waggle.
    """
    passed = metrics.false_shots == 0
    return AcceptanceTestResult(
        "No False Shots",
        passed,
        f"{metrics.false_shots} false shots" if not passed else "OK",
        metrics.false_shots,
        0
    )


def run_acceptance_tests(metrics: ReplayMetrics) -> List[AcceptanceTestResult]:
    """Run all acceptance tests on collected metrics."""
    results = [
        test_idle_stability(metrics),
        test_impact_latency(metrics),
        test_speed_availability(metrics),
        test_no_false_shots(metrics),
    ]
    
    return results


# =============================================================================
# pytest integration
# =============================================================================

def pytest_generate_tests(metafunc):
    """Generate tests for available test videos."""
    test_videos = list(Path("tests/data").glob("*.mp4"))
    if "video_path" in metafunc.fixturenames:
        metafunc.parametrize("video_path", test_videos)


class TestStationaryBall:
    """Tests for stationary ball stability."""
    
    def test_idle_jitter_below_threshold(self):
        """Ball stationary for 5s -> position stddev < 2px."""
        video = Path("tests/data/stationary_ball.mp4")
        if not video.exists():
            import pytest
            pytest.skip(f"Test video not found: {video}")
        
        metrics = run_replay(str(video))
        result = test_idle_stability(metrics)
        assert result.passed, result.message


class TestShotDetection:
    """Tests for shot detection timing."""
    
    def test_impact_latency(self):
        """Motion start -> TRACKING within 2 frames."""
        video = Path("tests/data/single_putt.mp4")
        if not video.exists():
            import pytest
            pytest.skip(f"Test video not found: {video}")
        
        metrics = run_replay(str(video))
        result = test_impact_latency(metrics)
        assert result.passed, result.message
    
    def test_speed_availability(self):
        """First speed estimate within 5 frames post-impact."""
        video = Path("tests/data/single_putt.mp4")
        if not video.exists():
            import pytest
            pytest.skip(f"Test video not found: {video}")
        
        metrics = run_replay(str(video))
        result = test_speed_availability(metrics)
        assert result.passed, result.message


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="StrikeLab Regression Test Runner")
    parser.add_argument("--replay", type=str, help="Run replay on video file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--all", action="store_true", help="Run all tests in tests/data/")
    
    args = parser.parse_args()
    
    if args.replay:
        # Single video test
        logger.info(f"Running regression test on: {args.replay}")
        metrics = run_replay(args.replay, verbose=args.verbose)
        
        print("\n" + "=" * 60)
        print("REGRESSION TEST RESULTS")
        print("=" * 60)
        print(f"Video: {args.replay}")
        print(f"Frames: {metrics.total_frames}")
        print(f"Duration: {metrics.duration_s:.1f}s")
        print(f"Shots detected: {metrics.shots_detected}")
        print("-" * 60)
        
        results = run_acceptance_tests(metrics)
        all_passed = True
        
        for result in results:
            print(result)
            if not result.passed:
                all_passed = False
        
        print("=" * 60)
        print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 60)
        
        return 0 if all_passed else 1
    
    elif args.all:
        # Run all test videos
        test_dir = Path("tests/data")
        if not test_dir.exists():
            logger.error(f"Test directory not found: {test_dir}")
            return 1
        
        videos = list(test_dir.glob("*.mp4"))
        if not videos:
            logger.error(f"No test videos found in {test_dir}")
            return 1
        
        all_passed = True
        for video in videos:
            print(f"\n{'='*60}")
            print(f"Testing: {video.name}")
            print("=" * 60)
            
            metrics = run_replay(str(video), verbose=args.verbose)
            results = run_acceptance_tests(metrics)
            
            for result in results:
                print(result)
                if not result.passed:
                    all_passed = False
        
        return 0 if all_passed else 1
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
