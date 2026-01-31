"""
Two-lane ball tracker with shot state machine for StrikeLab Putting Sim.

IDLE LANE: Strong stillness lock, conservative updates, minimal jitter
MOTION LANE: Fast tracking, minimal smoothing for accurate initial velocity

Acceptance metrics targets:
- Idle stability: < 2px stddev over 5 seconds
- Impact latency: ≤ 2 frames to TRACKING state
- Speed availability: ≤ 5 frames to first stable speed
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from .detector import Detection, BallDetector

logger = logging.getLogger(__name__)


class ShotState(Enum):
    """Shot detection state machine states."""
    ARMED = "ARMED"                    # Waiting for ball, ready to detect motion
    TRACKING = "TRACKING"              # Ball in motion, recording trajectory
    LOST_TRACK = "LOST_TRACK"          # Detection lost during tracking (occlusion/noise)
    VIRTUAL_ROLLING = "VIRTUAL_ROLLING"  # Ball exited frame, simulating virtually
    STOPPED = "STOPPED"                # Ball stopped, computing final metrics
    COOLDOWN = "COOLDOWN"              # Brief pause before re-arming


class BackgroundModel:
    """
    Simple background model for motion detection.
    
    Maintains a running average of frames when ball is stationary.
    Used to detect motion by computing foreground delta.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self._background: Optional[np.ndarray] = None
        self._update_count = 0
        
    def update(self, frame: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Update background model with new frame.
        
        Args:
            frame: BGR frame
            mask: Optional mask (255 = include in update, 0 = exclude)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        if self._background is None:
            self._background = gray.copy()
            self._update_count = 1
            return
        
        # Adaptive learning rate - faster initially, slower once stable
        alpha = min(self.learning_rate * 10, 0.5) if self._update_count < 30 else self.learning_rate
        
        if mask is not None:
            # Only update where mask is non-zero
            update_mask = mask > 0
            self._background[update_mask] = (
                alpha * gray[update_mask] + 
                (1 - alpha) * self._background[update_mask]
            )
        else:
            self._background = alpha * gray + (1 - alpha) * self._background
        
        self._update_count += 1
    
    def get_foreground_delta(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute foreground delta (difference from background).
        
        Returns:
            (delta_image, mean_delta) - delta image and mean absolute difference
        """
        if self._background is None:
            return np.zeros_like(frame[:, :, 0]), 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        delta = np.abs(gray - self._background)
        
        return delta.astype(np.uint8), float(np.mean(delta))
    
    def get_motion_mask(self, frame: np.ndarray, threshold: float = 25.0) -> np.ndarray:
        """Get binary mask of motion regions."""
        delta, _ = self.get_foreground_delta(frame)
        _, mask = cv2.threshold(delta, threshold, 255, cv2.THRESH_BINARY)
        return mask
    
    def reset(self):
        """Reset background model."""
        self._background = None
        self._update_count = 0
    
    @property
    def is_initialized(self) -> bool:
        return self._background is not None and self._update_count > 10


class TrackerLane(Enum):
    """Which tracking lane is active."""
    IDLE = "IDLE"
    MOTION = "MOTION"


@dataclass
class Velocity:
    """Velocity vector in pixel space."""
    vx: float  # pixels per second
    vy: float  # pixels per second
    
    @property
    def speed(self) -> float:
        """Speed magnitude in pixels per second."""
        return np.sqrt(self.vx**2 + self.vy**2)
    
    @property
    def direction_deg(self) -> float:
        """Direction in degrees (0 = right, 90 = down)."""
        return np.degrees(np.arctan2(self.vy, self.vx))
    
    def as_tuple(self) -> Tuple[float, float]:
        return (self.vx, self.vy)


@dataclass
class TrackPoint:
    """Single point in trajectory."""
    x: float
    y: float
    timestamp_ns: int
    frame_id: int
    confidence: float
    

@dataclass
class ExitState:
    """Ball state when exiting the camera frame."""
    position: Tuple[float, float]      # Exit position (x, y) in pixels
    velocity: Tuple[float, float]      # Exit velocity (vx, vy) in px/s
    speed: float                        # Exit speed in px/s
    direction_deg: float                # Exit direction in degrees
    curvature: float                    # Trajectory curvature (positive = curving right)
    timestamp_ns: int                   # When ball exited
    frame_id: int                       # Frame when ball exited
    trajectory_before_exit: List[Tuple[float, float]]  # Trajectory up to exit


@dataclass
class VirtualBallState:
    """State of the virtual ball after exiting frame."""
    x: float                            # Current virtual x position
    y: float                            # Current virtual y position
    vx: float                           # Current velocity x
    vy: float                           # Current velocity y
    speed: float                        # Current speed
    distance_traveled: float            # Total distance from exit point
    time_since_exit: float              # Time since ball exited (seconds)
    is_rolling: bool                    # Whether ball is still rolling
    final_position: Optional[Tuple[float, float]] = None  # Where ball will stop


@dataclass
class ShotResult:
    """Final shot metrics after ball stops."""
    initial_speed_px_s: float
    initial_direction_deg: float
    frames_to_tracking: int  # Frames from motion start to TRACKING state
    frames_to_speed: int     # Frames from impact to first stable speed
    trajectory: List[Tuple[float, float]]
    duration_ms: float
    physical_distance_px: float = 0.0  # Distance traveled in camera view
    virtual_distance_px: float = 0.0   # Distance traveled virtually (after frame exit)
    total_distance_px: float = 0.0     # Total distance (physical + virtual)
    exited_frame: bool = False         # Whether ball exited the camera view
    # Frozen calibration value used for this shot (for consistent distance reporting)
    pixels_per_meter: float = 1150.0
    

@dataclass
class TrackerState:
    """Current tracker state for external consumption."""
    state: ShotState
    lane: TrackerLane
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    ball_radius: Optional[float] = None
    ball_confidence: Optional[float] = None
    velocity: Optional[Velocity] = None
    shot_result: Optional[ShotResult] = None
    idle_stddev: float = 0.0  # Position jitter when idle
    virtual_ball: Optional[VirtualBallState] = None  # Virtual ball when rolling off-frame
    exit_state: Optional[ExitState] = None  # State when ball exited frame


@dataclass 
class FrameTimingStats:
    """Statistics about frame timing during a shot."""
    effective_fps: float = 0.0
    dt_mean_ms: float = 0.0
    dt_std_ms: float = 0.0
    dt_min_ms: float = 0.0
    dt_max_ms: float = 0.0
    frame_count: int = 0


@dataclass
class RobustVelocityEstimate:
    """Result of robust velocity estimation via regression."""
    vx: float                    # px/s
    vy: float                    # px/s
    speed: float                 # px/s magnitude
    direction_deg: float         # degrees
    r_squared: float             # fit quality [0,1]
    num_frames: int              # frames used in fit
    residual_mean: float         # mean residual in pixels
    source_frame_range: Tuple[int, int]  # (start_frame, end_frame)
    
    def is_trustworthy(self) -> bool:
        """Check if this estimate meets quality gates."""
        return (self.num_frames >= 6 and 
                self.r_squared >= 0.85 and 
                self.residual_mean <= 3.0)


@dataclass
class DistanceEstimate:
    """A candidate distance estimate with confidence."""
    total_px: float
    virtual_px: float
    method: str  # "trajectory_fit", "exit_velocity", "v0_robust", "physical_only"
    confidence: float  # [0, 1]
    details: dict  # diagnostic info


@dataclass
class StateTimelineEntry:
    """Entry in the state timeline for debugging."""
    timestamp_ns: int
    frame_id: int
    state: str
    event: str  # "transition", "v0_captured", "exit_detected", etc.
    details: dict  # speed, position, etc.


class BallTracker:
    """
    Two-lane ball tracker with shot state machine.
    
    IDLE LANE (when ball stationary):
    - Strong temporal smoothing (EMA with alpha=0.05)
    - Position only updates when stable for N frames
    - Maintains background model for motion detection
    - Goal: < 2px stddev over 5 seconds
    
    MOTION LANE (when ball moving):
    - ROI-based tracking centered on last known position
    - No smoothing for first 10 frames (preserve true motion)
    - Fast velocity computation from raw positions
    - Goal: First speed estimate within 5 frames
    
    Impact trigger:
    - Centroid displacement > threshold
    - OR background foreground delta spike
    - Debounced with 2 consecutive trigger frames
    """
    
    # State machine thresholds
    MOTION_THRESHOLD_PX = 25.0      # Displacement to trigger motion (higher = less false triggers)
    MOTION_CONFIRM_FRAMES = 4       # Consecutive frames to confirm motion
    STOPPED_VELOCITY_THRESHOLD = 50  # px/s to consider stopped
    STOPPED_CONFIRM_FRAMES = 10     # Frames at low velocity to confirm stop
    COOLDOWN_DURATION_MS = 500      # Cooldown before re-arming
    
    # Settling period - ignore motion detection while ball stabilizes
    SETTLING_FRAMES = 60            # ~0.5 seconds at 120fps to let ball settle after placement
    
    # IDLE lane parameters - tuned for butter-smooth idle
    IDLE_EMA_ALPHA = 0.01           # Very strong smoothing when idle (lower = smoother)
    IDLE_STABILITY_FRAMES = 5       # Frames before locking position
    IDLE_LOCK_THRESHOLD_PX = 3.0    # If movement < this, lock position completely
    
    # MOTION lane parameters
    MOTION_RAW_FRAMES = 10          # Frames without smoothing after impact
    VELOCITY_WINDOW = 5             # Frames for initial velocity computation (more samples)
    VELOCITY_WINDOW_STABLE = 8      # Frames for stable velocity computation
    EXIT_VELOCITY_WINDOW = 12       # Frames for exit velocity calculation (even more for accuracy)
    ROI_PADDING = 100               # Pixels to pad around ball for ROI tracking
    
    # Background model parameters
    BG_DELTA_THRESHOLD = 30.0       # Foreground delta to trigger motion
    BG_LEARNING_RATE = 0.01         # Background update rate
    
    # False shot prevention
    MAX_RADIUS_CHANGE_RATIO = 0.5   # Reject if radius changes > 50%
    MIN_CONFIDENCE_THRESHOLD = 0.7  # Minimum detection confidence
    MAX_POSITION_JUMP_PX = 100      # Large jump = ball placement, not shot
    MIN_SHOT_FRAMES = 5             # Minimum frames for valid shot
    MIN_SHOT_DISTANCE_PX = 50       # Minimum physical distance for valid shot (rejects false triggers)
    
    # Frame exit detection - for virtual ball continuation
    FRAME_EXIT_MARGIN_PX = 30       # Ball considered "exited" when this close to edge
    MIN_EXIT_SPEED_PX_S = 100       # Minimum speed to trigger virtual rolling
    MIN_TRACKING_FRAMES_FOR_EXIT = 10  # Need enough data for curve estimation
    
    # Motion direction filter - prevents false triggers from putter swing/hand movement
    VALID_MOTION_ANGLE_DEG = 45.0   # Accept motion within +/- this angle from forward
    FORWARD_DIRECTION_DEG = 0.0     # Default forward direction (0 = right, updated from calibration)
    
    # Virtual ball physics - LINEAR DECELERATION MODEL
    # Real putting physics: constant friction force → constant deceleration
    # v(t) = v0 - a*t, stops when v=0, distance = v0²/(2a)
    # 
    # Default deceleration: 0.55 m/s² (configurable via config.json)
    # Stored in m/s² and converted to px/s² using current calibration
    DECELERATION_M_S2 = 0.55        # Default deceleration in m/s²
    MIN_VIRTUAL_SPEED_PX_S = 20     # Stop virtual rolling below this speed
    MAX_VIRTUAL_TIME_S = 10.0       # Maximum virtual rolling time (safety limit)
    
    # Ball validation for motion trigger
    MIN_BALL_RADIUS_PX = 12         # Minimum ball radius for valid detection
    MAX_BALL_RADIUS_PX = 50         # Maximum ball radius
    MIN_CIRCULARITY = 0.80          # Minimum circularity to be considered a ball
    BALL_STABLE_FRAMES = 3          # Frames ball must be stable before triggering
    
    # Current calibration for pixel/meter conversion
    _current_pixels_per_meter: float = 1150.0  # Updated by set_calibration()
    
    # Effective FPS (from actual frame timestamps)
    _effective_fps: float = 100.0   # Updated from actual frame timing
    
    def __init__(
        self, 
        detector: Optional[BallDetector] = None,
        motion_threshold_px: Optional[float] = None,
        motion_confirm_frames: Optional[int] = None,
        stopped_velocity_threshold: Optional[float] = None,
        stopped_confirm_frames: Optional[int] = None,
        stopped_confirm_time_ms: Optional[int] = None,
        cooldown_duration_ms: Optional[int] = None,
        idle_ema_alpha: Optional[float] = None,
        deceleration_px_s2: Optional[float] = None,
        valid_motion_angle_deg: Optional[float] = None,
        forward_direction_deg: Optional[float] = None
    ):
        self._state = ShotState.ARMED
        self._lane = TrackerLane.IDLE
        
        # Optional detector for ROI-based tracking
        self._detector = detector
        
        # Apply config overrides (if provided, use them; otherwise use class constants)
        if motion_threshold_px is not None:
            self.MOTION_THRESHOLD_PX = motion_threshold_px
        if motion_confirm_frames is not None:
            self.MOTION_CONFIRM_FRAMES = motion_confirm_frames
        if stopped_velocity_threshold is not None:
            self.STOPPED_VELOCITY_THRESHOLD = stopped_velocity_threshold
        if stopped_confirm_frames is not None:
            self.STOPPED_CONFIRM_FRAMES = stopped_confirm_frames
        if cooldown_duration_ms is not None:
            self.COOLDOWN_DURATION_MS = cooldown_duration_ms
        if idle_ema_alpha is not None:
            self.IDLE_EMA_ALPHA = idle_ema_alpha
        if deceleration_px_s2 is not None:
            # If passed in px/s², convert to m/s² using default ppm
            self.DECELERATION_M_S2 = deceleration_px_s2 / 1150.0
            logger.info(f"Virtual ball deceleration set to {self.DECELERATION_M_S2:.3f} m/s²")
        if valid_motion_angle_deg is not None:
            self.VALID_MOTION_ANGLE_DEG = valid_motion_angle_deg
        if forward_direction_deg is not None:
            self.FORWARD_DIRECTION_DEG = forward_direction_deg
        
        # Timestamp-based stop detection (default 100ms)
        self._stopped_confirm_time_ms = stopped_confirm_time_ms if stopped_confirm_time_ms is not None else 100
        
        # Motion direction filter state
        self._forward_direction_deg = self.FORWARD_DIRECTION_DEG
        self._valid_motion_angle_deg = self.VALID_MOTION_ANGLE_DEG
        
        # Position tracking
        self._current_pos: Optional[Tuple[float, float]] = None
        self._smoothed_pos: Optional[Tuple[float, float]] = None
        self._locked_pos: Optional[Tuple[float, float]] = None  # Stillness lock position
        self._last_radius: Optional[float] = None
        self._last_confidence: float = 0.0
        
        # Trajectory recording
        self._trajectory: deque[TrackPoint] = deque(maxlen=1000)
        self._raw_trajectory: deque[TrackPoint] = deque(maxlen=100)  # Raw positions for velocity & physics fitting
        self._motion_start_frame: int = 0
        self._first_speed_frame: int = 0
        
        # Frame timing tracking for accurate dt calculation
        self._frame_timestamps: deque[int] = deque(maxlen=120)  # Last ~1 second of timestamps
        self._shot_timestamps: List[int] = []  # Timestamps during current shot
        
        # Motion detection
        self._motion_trigger_count = 0
        self._stopped_count = 0
        self._cooldown_start_ns: int = 0
        self._settling_countdown: int = 0  # Frames to wait before motion detection
        
        # Ball validation for trigger
        self._ball_stable_count: int = 0  # Consecutive frames with valid ball detection
        self._last_ball_check_radius: float = 0.0
        
        # Background model for motion detection
        self._background = BackgroundModel(learning_rate=self.BG_LEARNING_RATE)
        self._last_frame: Optional[np.ndarray] = None
        self._foreground_delta: float = 0.0
        
        # Idle stability tracking  
        self._idle_positions: deque[Tuple[float, float]] = deque(maxlen=600)  # ~5s at 120fps
        self._idle_stability_count = 0
        self._idle_stddev_history: deque[float] = deque(maxlen=30)
        
        # Velocity computation
        self._velocity: Optional[Velocity] = None
        self._velocity_history: deque[Velocity] = deque(maxlen=10)
        
        # Shot result
        self._shot_result: Optional[ShotResult] = None
        self._impact_velocity: Optional[Velocity] = None
        self._shot_start_pos: Optional[Tuple[float, float]] = None  # Position where shot started
        
        # ROI for motion lane tracking
        self._roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
        
        # Frame bounds for exit detection (set externally based on camera resolution)
        self._frame_width: int = 1280
        self._frame_height: int = 800
        
        # Exit detection and virtual ball state
        self._exit_state: Optional[ExitState] = None
        self._virtual_ball: Optional[VirtualBallState] = None
        self._virtual_start_time_ns: int = 0
        self._frames_lost: int = 0  # Consecutive frames without detection during tracking
        self._last_valid_position: Optional[Tuple[float, float]] = None
        self._last_valid_velocity: Optional[Velocity] = None
        
        # Fitted physics for more accurate virtual rolling
        self._fitted_deceleration: Optional[float] = None  # px/s² from trajectory fit
        self._fitted_physics: Optional[Tuple[float, float, float]] = None  # (v0, a, total_dist)
        
        # Expected distances calculated from initial velocity (more reliable than exit velocity)
        self._expected_total_distance_px: float = 0.0
        self._expected_virtual_distance_px: float = 0.0
        
        # Shot timing stats
        self._shot_timing_stats: Optional[FrameTimingStats] = None
        
        # State timeline for debugging
        self._state_timeline: List[StateTimelineEntry] = []
        
        # Robust v0 estimate (computed via regression on frames [motion_start+2..motion_start+10])
        self._robust_v0: Optional[RobustVelocityEstimate] = None
        
        # Timestamp-based stop detection
        self._low_velocity_start_ns: Optional[int] = None
        
        # LOST_TRACK state tracking
        self._lost_track_count: int = 0
        
    def update(
        self, 
        detection: Optional[Detection], 
        timestamp_ns: int, 
        frame_id: int,
        frame: Optional[np.ndarray] = None
    ) -> TrackerState:
        """
        Update tracker with new detection.
        
        Args:
            detection: Ball detection (or None if not found)
            timestamp_ns: Frame timestamp in nanoseconds
            frame_id: Frame number
            frame: Optional BGR frame for background model update
            
        Returns:
            Current tracker state
        """
        # Track frame timestamps for FPS calculation
        self._frame_timestamps.append(timestamp_ns)
        self._update_effective_fps()
        
        # Track timestamps during shot
        if self._state in (ShotState.TRACKING, ShotState.VIRTUAL_ROLLING):
            self._shot_timestamps.append(timestamp_ns)
        
        # Update frame bounds from frame if available
        if frame is not None:
            h, w = frame.shape[:2]
            if w != self._frame_width or h != self._frame_height:
                self.set_frame_bounds(w, h)
        
        # Update background model if frame provided and in IDLE
        if frame is not None:
            self._last_frame = frame
            if self._lane == TrackerLane.IDLE and detection is not None:
                # Create mask excluding ball area for background update
                mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
                if self._current_pos is not None:
                    radius = int((self._last_radius or 15) * 2)
                    cv2.circle(
                        mask, 
                        (int(self._current_pos[0]), int(self._current_pos[1])),
                        radius, 0, -1
                    )
                self._background.update(frame, mask)
            
            # Compute foreground delta for motion detection
            if self._background.is_initialized:
                _, self._foreground_delta = self._background.get_foreground_delta(frame)
        
        # Handle VIRTUAL_ROLLING state - update virtual ball physics
        if self._state == ShotState.VIRTUAL_ROLLING:
            self._update_virtual_ball(timestamp_ns, frame_id)
            return self._build_state()
        
        # Handle state-specific logic
        if self._state == ShotState.COOLDOWN:
            self._handle_cooldown(timestamp_ns)
        
        # Check for frame exit during tracking (even without detection)
        if self._state == ShotState.TRACKING:
            if self._check_frame_exit(detection, timestamp_ns, frame_id):
                self._transition_to_virtual_rolling(timestamp_ns, frame_id)
                return self._build_state()
        
        if detection is None:
            return self._build_state()
        
        # Validate detection
        if not self._validate_detection(detection):
            return self._build_state()
        
        # Update position based on current lane
        if self._lane == TrackerLane.IDLE:
            self._update_idle_lane(detection, timestamp_ns, frame_id)
        else:
            self._update_motion_lane(detection, timestamp_ns, frame_id)
        
        # Check for state transitions
        self._check_transitions(detection, timestamp_ns, frame_id)
        
        return self._build_state()
    
    def _update_effective_fps(self):
        """Update effective FPS from actual frame timestamps."""
        if len(self._frame_timestamps) < 10:
            return
        
        timestamps = list(self._frame_timestamps)
        dt_ns = timestamps[-1] - timestamps[0]
        if dt_ns > 0:
            self._effective_fps = (len(timestamps) - 1) / (dt_ns / 1e9)
    
    def _get_dt_from_timestamps(self, t1_ns: int, t2_ns: int) -> float:
        """Get time delta in seconds from two timestamps."""
        dt_ns = t2_ns - t1_ns
        return dt_ns / 1e9 if dt_ns > 0 else (1.0 / self._effective_fps)
    
    def _get_expected_frame_dt(self) -> float:
        """Get expected dt for one frame based on effective FPS."""
        return 1.0 / max(self._effective_fps, 30.0)
    
    def _validate_detection(self, detection: Detection) -> bool:
        """Validate detection to prevent false shots."""
        # Confidence check
        if detection.confidence < self.MIN_CONFIDENCE_THRESHOLD:
            logger.debug(f"Low confidence detection: {detection.confidence:.2f}")
            return False
        
        # Radius change check (hand occlusion)
        if self._last_radius is not None:
            ratio = detection.radius / self._last_radius
            if ratio < (1 - self.MAX_RADIUS_CHANGE_RATIO) or ratio > (1 + self.MAX_RADIUS_CHANGE_RATIO):
                logger.debug(f"Large radius change: {self._last_radius:.1f} -> {detection.radius:.1f}")
                # Don't reject, but don't update radius
                return True
        
        self._last_radius = detection.radius
        self._last_confidence = detection.confidence
        return True
    
    def _validate_ball_signature(self, detection: Detection) -> bool:
        """
        Validate that detection matches ball signature (not putter/hand).
        
        Returns True if detection looks like a golf ball.
        """
        # Radius check
        if detection.radius < self.MIN_BALL_RADIUS_PX or detection.radius > self.MAX_BALL_RADIUS_PX:
            return False
        
        # Confidence check (stricter for trigger)
        if detection.confidence < 0.85:
            return False
        
        # Radius stability check
        if self._last_ball_check_radius > 0:
            ratio = detection.radius / self._last_ball_check_radius
            if ratio < 0.7 or ratio > 1.3:
                self._ball_stable_count = 0
                self._last_ball_check_radius = detection.radius
                return False
        
        self._last_ball_check_radius = detection.radius
        self._ball_stable_count += 1
        
        return self._ball_stable_count >= self.BALL_STABLE_FRAMES
    
    def _update_idle_lane(self, detection: Detection, timestamp_ns: int, frame_id: int):
        """
        Update position using IDLE lane (strong smoothing + stillness lock).
        
        The stillness lock ensures near-zero jitter when ball is stationary:
        1. Strong EMA smoothing on position
        2. If movement < threshold for N frames, lock position completely
        3. Only unlock when significant motion detected
        """
        new_pos = (detection.cx, detection.cy)
        
        if self._smoothed_pos is None:
            self._smoothed_pos = new_pos
            self._locked_pos = new_pos
            self._current_pos = new_pos
            self._idle_stability_count = 0
        else:
            # Check for large position jump (ball placement)
            dist = np.sqrt(
                (new_pos[0] - self._smoothed_pos[0])**2 +
                (new_pos[1] - self._smoothed_pos[1])**2
            )
            
            if dist > self.MAX_POSITION_JUMP_PX:
                # Ball was placed - reset to new position and start settling period
                logger.info(f"Ball placement detected (jump={dist:.1f}px), settling for {self.SETTLING_FRAMES} frames")
                self._smoothed_pos = new_pos
                self._locked_pos = new_pos
                self._current_pos = new_pos
                self._idle_positions.clear()
                self._idle_stability_count = 0
                self._settling_countdown = self.SETTLING_FRAMES  # Start settling period
                self._background.reset()  # Reset background for new ball position
                self._ball_stable_count = 0  # Reset ball validation
                return
            
            # EMA smoothing
            alpha = self.IDLE_EMA_ALPHA
            self._smoothed_pos = (
                alpha * new_pos[0] + (1 - alpha) * self._smoothed_pos[0],
                alpha * new_pos[1] + (1 - alpha) * self._smoothed_pos[1]
            )
            
            # Check if movement is below stillness threshold
            if self._locked_pos is not None:
                dist_from_lock = np.sqrt(
                    (self._smoothed_pos[0] - self._locked_pos[0])**2 +
                    (self._smoothed_pos[1] - self._locked_pos[1])**2
                )
                
                if dist_from_lock < self.IDLE_LOCK_THRESHOLD_PX:
                    self._idle_stability_count += 1
                    # Lock position after stability frames
                    if self._idle_stability_count >= self.IDLE_STABILITY_FRAMES:
                        # Position is locked - output locked position for near-zero jitter
                        self._current_pos = self._locked_pos
                    else:
                        self._current_pos = self._smoothed_pos
                else:
                    # Movement detected - update lock position
                    self._idle_stability_count = 0
                    self._locked_pos = self._smoothed_pos
                    self._current_pos = self._smoothed_pos
            else:
                self._locked_pos = self._smoothed_pos
                self._current_pos = self._smoothed_pos
        
        # Track idle positions for jitter measurement
        self._idle_positions.append((detection.cx, detection.cy))  # Raw positions for stddev
        
        # Record trajectory point (use smoothed/locked position)
        self._trajectory.append(TrackPoint(
            x=self._current_pos[0],
            y=self._current_pos[1],
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            confidence=detection.confidence
        ))
    
    def _update_motion_lane(self, detection: Detection, timestamp_ns: int, frame_id: int):
        """
        Update position using MOTION lane (minimal smoothing for fast response).
        
        Key for acceptance metrics:
        - No smoothing for first MOTION_RAW_FRAMES frames
        - Velocity computed from raw positions for accuracy
        - ROI updated for efficient tracking
        """
        new_pos = (detection.cx, detection.cy)
        
        # Check for suspicious position jump during tracking (might be noise after ball exits)
        if self._current_pos and self._velocity:
            dx = new_pos[0] - self._current_pos[0]
            dy = new_pos[1] - self._current_pos[1]
            jump_dist = np.sqrt(dx**2 + dy**2)
            
            # Expected movement based on velocity and ACTUAL frame time
            expected_dt = self._get_expected_frame_dt()
            expected_move = self._velocity.speed * expected_dt
            
            # If jump is way larger than expected, this might be noise
            if jump_dist > expected_move * 5 and jump_dist > 50:
                logger.warning(f"Suspicious position jump during tracking: {jump_dist:.1f}px "
                             f"(expected ~{expected_move:.1f}px). Treating as lost frame.")
                self._frames_lost += 1
                # Don't update position with this suspicious detection
                return
        
        # For first N frames after impact, use raw position (critical for speed accuracy)
        frames_since_motion = frame_id - self._motion_start_frame
        
        if frames_since_motion <= self.MOTION_RAW_FRAMES:
            # Raw position - no smoothing at all
            self._current_pos = new_pos
        else:
            # Light smoothing after initial burst
            if self._current_pos:
                alpha = 0.8  # Very light smoothing
                self._current_pos = (
                    alpha * new_pos[0] + (1 - alpha) * self._current_pos[0],
                    alpha * new_pos[1] + (1 - alpha) * self._current_pos[1]
                )
            else:
                self._current_pos = new_pos
        
        # Update ROI for next frame (centered on ball with padding)
        self._update_roi(detection)
        
        # Record raw trajectory point for velocity computation
        self._raw_trajectory.append(TrackPoint(
            x=new_pos[0],  # Always raw position
            y=new_pos[1],
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            confidence=detection.confidence
        ))
        
        # Record trajectory (may be smoothed)
        self._trajectory.append(TrackPoint(
            x=self._current_pos[0],
            y=self._current_pos[1],
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            confidence=detection.confidence
        ))
        
        # Compute velocity from raw positions
        self._compute_velocity_fast()
        
        # Record first stable speed (target: ≤5 frames)
        if self._velocity and self._velocity.speed > self.STOPPED_VELOCITY_THRESHOLD:
            if self._first_speed_frame == 0:
                frames_to_speed = frame_id - self._motion_start_frame
                self._first_speed_frame = frame_id
                self._impact_velocity = self._velocity
                logger.info(
                    f"First speed at frame {frame_id} ({frames_to_speed} frames): "
                    f"{self._velocity.speed:.1f} px/s, dir={self._velocity.direction_deg:.1f}°"
                )
    
    def _update_roi(self, detection: Detection):
        """Update ROI for next frame's detection."""
        if self._last_frame is None:
            return
        
        h, w = self._last_frame.shape[:2]
        cx, cy = int(detection.cx), int(detection.cy)
        radius = int(detection.radius or 15)
        
        # ROI is ball position + padding, clamped to frame bounds
        pad = self.ROI_PADDING
        x1 = max(0, cx - pad - radius)
        y1 = max(0, cy - pad - radius)
        x2 = min(w, cx + pad + radius)
        y2 = min(h, cy + pad + radius)
        
        self._roi = (x1, y1, x2 - x1, y2 - y1)
    
    def _compute_velocity_fast(self):
        """
        Compute velocity from raw trajectory points using actual timestamps.
        Uses smaller window for faster initial response.
        """
        if len(self._raw_trajectory) < 2:
            self._velocity = None
            return
        
        # Use smaller window initially for fast response
        frames_since_motion = len(self._raw_trajectory)
        window = self.VELOCITY_WINDOW if frames_since_motion <= 10 else self.VELOCITY_WINDOW_STABLE
        
        points = list(self._raw_trajectory)[-window:]
        if len(points) < 2:
            return
        
        # Use actual timestamps for velocity calculation
        p1 = points[0]
        p2 = points[-1]
        
        dt_s = self._get_dt_from_timestamps(p1.timestamp_ns, p2.timestamp_ns)
        if dt_s <= 0:
            return
        
        vx = (p2.x - p1.x) / dt_s
        vy = (p2.y - p1.y) / dt_s
        
        self._velocity = Velocity(vx=vx, vy=vy)
        self._velocity_history.append(self._velocity)
    
    def _compute_velocity(self):
        """Compute velocity from trajectory points (legacy, used for stable tracking)."""
        if len(self._trajectory) < 2:
            self._velocity = None
            return
        
        points = list(self._trajectory)[-self.VELOCITY_WINDOW_STABLE:]
        if len(points) < 2:
            return
        
        p1 = points[0]
        p2 = points[-1]
        
        dt_s = self._get_dt_from_timestamps(p1.timestamp_ns, p2.timestamp_ns)
        if dt_s <= 0:
            return
        
        vx = (p2.x - p1.x) / dt_s
        vy = (p2.y - p1.y) / dt_s
        
        self._velocity = Velocity(vx=vx, vy=vy)
        self._velocity_history.append(self._velocity)
    
    def _compute_exit_velocity_regression(self) -> Optional[Tuple[Velocity, float]]:
        """
        Compute exit velocity using linear regression on x(t), y(t).
        
        More robust than weighted average - fits a line to positions vs time
        and uses the slope as velocity. Includes outlier rejection.
        
        Returns:
            (Velocity, r_squared) or (None, 0) if insufficient data
        """
        if len(self._raw_trajectory) < 8:
            return None, 0.0
        
        points = list(self._raw_trajectory)[-min(20, len(self._raw_trajectory)):]
        if len(points) < 5:
            return None, 0.0
        
        # Build arrays
        times = []
        xs = []
        ys = []
        
        t0 = points[0].timestamp_ns
        for p in points:
            t = (p.timestamp_ns - t0) / 1e9  # Convert to seconds
            times.append(t)
            xs.append(p.x)
            ys.append(p.y)
        
        times = np.array(times)
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Simple outlier rejection using median absolute deviation
        def reject_outliers(arr, threshold=2.5):
            """Returns mask of inliers."""
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            if mad < 1e-6:
                return np.ones(len(arr), dtype=bool)
            modified_z = 0.6745 * (arr - med) / mad
            return np.abs(modified_z) < threshold
        
        # Compute velocities between consecutive points for outlier detection
        velocities = []
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            if dt > 0:
                vx = (xs[i] - xs[i-1]) / dt
                vy = (ys[i] - ys[i-1]) / dt
                velocities.append(np.sqrt(vx**2 + vy**2))
            else:
                velocities.append(0)
        velocities = np.array([0] + velocities)  # Pad first element
        
        inlier_mask = reject_outliers(velocities)
        
        if np.sum(inlier_mask) < 5:
            inlier_mask = np.ones(len(times), dtype=bool)
        
        times_clean = times[inlier_mask]
        xs_clean = xs[inlier_mask]
        ys_clean = ys[inlier_mask]
        
        if len(times_clean) < 3:
            return None, 0.0
        
        # Linear regression: x = vx*t + x0, y = vy*t + y0
        n = len(times_clean)
        sum_t = np.sum(times_clean)
        sum_t2 = np.sum(times_clean**2)
        
        denom = n * sum_t2 - sum_t**2
        if abs(denom) < 1e-10:
            return None, 0.0
        
        # Fit x(t)
        sum_x = np.sum(xs_clean)
        sum_tx = np.sum(times_clean * xs_clean)
        vx = (n * sum_tx - sum_t * sum_x) / denom
        x0 = (sum_x - vx * sum_t) / n
        
        # Fit y(t)
        sum_y = np.sum(ys_clean)
        sum_ty = np.sum(times_clean * ys_clean)
        vy = (n * sum_ty - sum_t * sum_y) / denom
        y0 = (sum_y - vy * sum_t) / n
        
        # Calculate R² for x
        x_pred = vx * times_clean + x0
        ss_res_x = np.sum((xs_clean - x_pred)**2)
        ss_tot_x = np.sum((xs_clean - np.mean(xs_clean))**2)
        r2_x = 1 - (ss_res_x / ss_tot_x) if ss_tot_x > 0 else 0
        
        # Calculate R² for y
        y_pred = vy * times_clean + y0
        ss_res_y = np.sum((ys_clean - y_pred)**2)
        ss_tot_y = np.sum((ys_clean - np.mean(ys_clean))**2)
        r2_y = 1 - (ss_res_y / ss_tot_y) if ss_tot_y > 0 else 0
        
        # Combined R² (weighted by variance in each direction)
        total_var = ss_tot_x + ss_tot_y
        if total_var > 0:
            r2 = (r2_x * ss_tot_x + r2_y * ss_tot_y) / total_var
        else:
            r2 = 0.5 * (r2_x + r2_y)
        
        result = Velocity(vx=vx, vy=vy)
        
        logger.debug(f"Exit velocity (regression): {result.speed:.1f} px/s @ {result.direction_deg:.1f}°, "
                    f"R²={r2:.3f}, points={len(times_clean)}")
        
        return result, r2
    
    def _compute_exit_velocity(self) -> Optional[Velocity]:
        """
        Compute exit velocity using robust methods.
        
        Priority:
        1. Linear regression (if R² > 0.8)
        2. Weighted average (fallback)
        
        Returns:
            Velocity object or None if insufficient data
        """
        # Try regression first (more robust)
        reg_vel, r2 = self._compute_exit_velocity_regression()
        if reg_vel and r2 > 0.8:
            logger.info(f"Exit velocity (regression): {reg_vel.speed:.1f} px/s @ {reg_vel.direction_deg:.1f}°")
            return reg_vel
        
        # Fallback to weighted average
        if len(self._raw_trajectory) < self.EXIT_VELOCITY_WINDOW:
            return self._velocity
        
        points = list(self._raw_trajectory)[-self.EXIT_VELOCITY_WINDOW:]
        if len(points) < 5:
            return self._velocity
        
        # Calculate instantaneous velocities between consecutive points
        velocities = []
        for i in range(1, len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            
            dt_s = self._get_dt_from_timestamps(p1.timestamp_ns, p2.timestamp_ns)
            if dt_s <= 0:
                continue
            
            vx = (p2.x - p1.x) / dt_s
            vy = (p2.y - p1.y) / dt_s
            velocities.append((vx, vy))
        
        if len(velocities) < 3:
            return self._velocity
        
        # Exponential weights: more recent = higher weight
        weights = np.exp(np.linspace(-2, 0, len(velocities)))
        weights = weights / weights.sum()
        
        # Weighted average velocity
        vx_weighted = sum(w * v[0] for w, v in zip(weights, velocities))
        vy_weighted = sum(w * v[1] for w, v in zip(weights, velocities))
        
        result = Velocity(vx=vx_weighted, vy=vy_weighted)
        
        logger.info(f"Exit velocity (weighted): {result.speed:.1f} px/s @ {result.direction_deg:.1f}°")
        
        return result
    
    def _compute_robust_v0(self) -> Optional[RobustVelocityEstimate]:
        """
        Compute robust initial velocity via linear regression.
        
        Uses frames [motion_start+2 .. motion_start+10] to avoid:
        - First 2 frames (often contaminated by impact blur/detection jitter)
        - Single-frame noise
        
        Process:
        1. Collect positions and timestamps for frames in range
        2. Compute point-to-point velocities
        3. Reject outliers using MAD (median absolute deviation)
        4. Fit linear regression: x(t) = x0 + vx*t, y(t) = y0 + vy*t
        5. Return velocity with quality metrics
        
        Quality gates (all must pass for trustworthy estimate):
        - num_frames >= 6
        - R² >= 0.85
        - residual_mean <= 3.0 px
        """
        MIN_FRAMES = 6
        TARGET_RANGE = (2, 10)  # frames relative to motion_start
        
        # Get trajectory points in range
        points = [p for p in self._raw_trajectory 
                  if TARGET_RANGE[0] <= (p.frame_id - self._motion_start_frame) <= TARGET_RANGE[1]]
        
        if len(points) < MIN_FRAMES:
            logger.debug(f"Robust v0: insufficient points ({len(points)} < {MIN_FRAMES})")
            return None
        
        # Extract timestamps and positions
        t0 = points[0].timestamp_ns
        times = np.array([(p.timestamp_ns - t0) / 1e9 for p in points])
        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])
        
        # Compute point-to-point velocities for outlier detection
        velocities = []
        for i in range(1, len(points)):
            dt = times[i] - times[i-1]
            if dt > 0:
                vx = (xs[i] - xs[i-1]) / dt
                vy = (ys[i] - ys[i-1]) / dt
                velocities.append(np.sqrt(vx**2 + vy**2))
        
        if len(velocities) < 3:
            logger.debug(f"Robust v0: insufficient velocity samples ({len(velocities)} < 3)")
            return None
        
        # MAD-based outlier rejection
        velocities_arr = np.array(velocities)
        median_v = np.median(velocities_arr)
        mad = np.median(np.abs(velocities_arr - median_v))
        if mad < 1e-6:
            mad = np.std(velocities_arr) / 1.4826  # fallback to std
        threshold = median_v + 3 * mad * 1.4826  # 3-sigma equivalent
        
        # Build inlier mask
        inlier_mask = np.ones(len(points), dtype=bool)
        inlier_mask[0] = True  # Always keep first point
        for i in range(len(velocities)):
            if velocities[i] > threshold:
                inlier_mask[i + 1] = False
        
        # Re-extract with inliers only
        times_clean = times[inlier_mask]
        xs_clean = xs[inlier_mask]
        ys_clean = ys[inlier_mask]
        
        if len(times_clean) < MIN_FRAMES:
            logger.debug(f"Robust v0: too many outliers ({len(times_clean)} < {MIN_FRAMES} after rejection)")
            return None  # Too many outliers
        
        # Linear regression: position = p0 + v*t
        n = len(times_clean)
        sum_t = np.sum(times_clean)
        sum_t2 = np.sum(times_clean**2)
        denom = n * sum_t2 - sum_t**2
        
        if abs(denom) < 1e-10:
            return None
        
        # Fit x(t)
        sum_x = np.sum(xs_clean)
        sum_tx = np.sum(times_clean * xs_clean)
        vx = (n * sum_tx - sum_t * sum_x) / denom
        x0 = (sum_x - vx * sum_t) / n
        
        # Fit y(t)
        sum_y = np.sum(ys_clean)
        sum_ty = np.sum(times_clean * ys_clean)
        vy = (n * sum_ty - sum_t * sum_y) / denom
        y0 = (sum_y - vy * sum_t) / n
        
        # Compute R² and residuals
        x_pred = x0 + vx * times_clean
        y_pred = y0 + vy * times_clean
        
        ss_res = np.sum((xs_clean - x_pred)**2 + (ys_clean - y_pred)**2)
        ss_tot = np.sum((xs_clean - np.mean(xs_clean))**2 + (ys_clean - np.mean(ys_clean))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        residuals = np.sqrt((xs_clean - x_pred)**2 + (ys_clean - y_pred)**2)
        residual_mean = float(np.mean(residuals))
        
        speed = np.sqrt(vx**2 + vy**2)
        direction = np.degrees(np.arctan2(vy, vx))
        
        result = RobustVelocityEstimate(
            vx=float(vx),
            vy=float(vy),
            speed=float(speed),
            direction_deg=float(direction),
            r_squared=float(r_squared),
            num_frames=len(times_clean),
            residual_mean=residual_mean,
            source_frame_range=(points[0].frame_id, points[-1].frame_id)
        )
        
        logger.info(f"Robust v0: {speed:.1f} px/s @ {direction:.1f}°, R²={r_squared:.3f}, "
                   f"frames={len(times_clean)}, residual={residual_mean:.2f}px, "
                   f"trustworthy={result.is_trustworthy()}")
        
        return result
    
    def _compute_distance_estimate(
        self,
        physical_distance_px: float,
        trajectory_fit_result: Optional[Tuple[float, float, float]],
        exit_velocity: Optional[Velocity],
        exit_r_squared: float,
        robust_v0: Optional[RobustVelocityEstimate]
    ) -> DistanceEstimate:
        """
        Compute total distance using deterministic fallback strategy.
        
        PRIORITY ORDER (strict):
        1. Trajectory physics fit - if passes guardrails
        2. Exit velocity - PRIMARY fallback when fit rejected
           total = physical + (v_exit²)/(2a)
        3. Robust v0 regression - ONLY if exit confidence low/missing
           AND v0 passes quality gates (R² >= 0.85, frames >= 6, residual <= 3px)
        4. Physical only - last resort
        
        NEVER uses 2-frame v0.
        NEVER allows total < physical.
        """
        a = self.get_deceleration_px_s2()
        ppm = self._current_pixels_per_meter
        
        # Prepare diagnostic details
        details = {
            "physical_px": physical_distance_px,
            "physical_cm": (physical_distance_px / ppm) * 100,
            "deceleration_px_s2": a,
            "trajectory_fit_passed": trajectory_fit_result is not None,
            "exit_velocity_px_s": exit_velocity.speed if exit_velocity else 0,
            "exit_r_squared": exit_r_squared,
            "v0_robust_px_s": robust_v0.speed if robust_v0 else 0,
            "v0_robust_r_squared": robust_v0.r_squared if robust_v0 else 0,
            "v0_robust_frames": robust_v0.num_frames if robust_v0 else 0,
            "v0_robust_residual": robust_v0.residual_mean if robust_v0 else 0,
        }
        
        # === PRIORITY 1: Trajectory fit (best when it works) ===
        if trajectory_fit_result is not None:
            fitted_v0, fitted_a, fitted_total = trajectory_fit_result
            virtual_px = max(0, fitted_total - physical_distance_px)
            details["fitted_v0"] = fitted_v0
            details["fitted_a"] = fitted_a
            details["fitted_total_px"] = fitted_total
            logger.info(f"Distance estimate: TRAJECTORY_FIT - total={fitted_total/ppm*100:.1f}cm")
            return DistanceEstimate(
                total_px=fitted_total,
                virtual_px=virtual_px,
                method="trajectory_fit",
                confidence=0.95,
                details=details
            )
        
        # === PRIORITY 2: Exit velocity (PRIMARY fallback) ===
        # Use exit velocity if we have a reasonable measurement
        exit_confidence = 0.0
        d_total_exit = physical_distance_px  # default
        
        if exit_velocity and exit_velocity.speed > self.MIN_EXIT_SPEED_PX_S:
            virtual_from_exit = (exit_velocity.speed ** 2) / (2 * a)
            d_total_exit = physical_distance_px + virtual_from_exit
            
            # Compute exit confidence based on measurement quality
            # Higher R² from regression = more confident
            if exit_r_squared > 0.8:
                exit_confidence = 0.85
            elif exit_r_squared > 0.6:
                exit_confidence = 0.7
            elif exit_velocity.speed > self.MIN_EXIT_SPEED_PX_S * 2:
                # Reasonable speed even if fit quality lower
                exit_confidence = 0.5
            else:
                exit_confidence = 0.3
            
            details["d_total_exit_px"] = d_total_exit
            details["d_total_exit_cm"] = (d_total_exit / ppm) * 100
            details["exit_confidence"] = exit_confidence
        
        # === PRIORITY 3: Robust v0 (secondary fallback) ===
        # Only use if exit confidence is low AND v0 passes strict quality gates
        v0_confidence = 0.0
        d_total_v0 = physical_distance_px  # default
        
        if robust_v0 and robust_v0.speed > 0:
            d_total_v0 = (robust_v0.speed ** 2) / (2 * a)
            
            # Only trust v0 if it passes ALL quality gates
            if robust_v0.is_trustworthy():
                v0_confidence = 0.7
            else:
                v0_confidence = 0.0  # Don't use unreliable v0
            
            details["d_total_v0_px"] = d_total_v0
            details["d_total_v0_cm"] = (d_total_v0 / ppm) * 100
            details["v0_confidence"] = v0_confidence
            details["v0_trustworthy"] = robust_v0.is_trustworthy()
        
        # === Decision logic ===
        # Exit velocity is PRIMARY - use it unless confidence is very low
        if exit_confidence >= 0.5:
            # Good exit velocity - use it
            chosen_total = d_total_exit
            method = "exit_velocity"
            confidence = exit_confidence
            logger.info(f"Distance estimate: EXIT_VELOCITY - total={d_total_exit/ppm*100:.1f}cm (conf={exit_confidence:.2f})")
        elif exit_confidence > 0 and v0_confidence > 0:
            # Both available but exit is low confidence
            # Use v0 ONLY if it's trustworthy AND gives reasonable result
            if v0_confidence > exit_confidence and robust_v0 and robust_v0.is_trustworthy():
                # v0 is more reliable - but sanity check against exit
                # v0-based total should not be >50% higher than exit-based
                if d_total_v0 <= d_total_exit * 1.5:
                    chosen_total = d_total_v0
                    method = "v0_robust"
                    confidence = v0_confidence
                    logger.info(f"Distance estimate: V0_ROBUST - total={d_total_v0/ppm*100:.1f}cm (exit was low conf)")
                else:
                    # v0 seems inflated - fall back to exit
                    chosen_total = d_total_exit
                    method = "exit_velocity"
                    confidence = exit_confidence
                    logger.warning(f"Distance estimate: EXIT_VELOCITY (v0 seemed inflated: {d_total_v0/ppm*100:.1f} vs {d_total_exit/ppm*100:.1f})")
            else:
                # Exit is still best option despite low confidence
                chosen_total = d_total_exit
                method = "exit_velocity"
                confidence = exit_confidence
                logger.info(f"Distance estimate: EXIT_VELOCITY (low conf) - total={d_total_exit/ppm*100:.1f}cm")
        elif v0_confidence > 0 and robust_v0 and robust_v0.is_trustworthy():
            # No exit velocity but have trustworthy v0
            chosen_total = d_total_v0
            method = "v0_robust"
            confidence = v0_confidence
            logger.info(f"Distance estimate: V0_ROBUST (no exit) - total={d_total_v0/ppm*100:.1f}cm")
        else:
            # No reliable estimates - use physical only
            chosen_total = physical_distance_px
            method = "physical_only"
            confidence = 0.3
            logger.warning(f"Distance estimate: PHYSICAL_ONLY - {physical_distance_px/ppm*100:.1f}cm (no reliable velocity)")
        
        # NEVER allow total < physical (would mean negative virtual)
        chosen_total = max(chosen_total, physical_distance_px)
        virtual_px = chosen_total - physical_distance_px
        
        details["chosen_method"] = method
        details["chosen_total_px"] = chosen_total
        details["chosen_total_cm"] = (chosen_total / ppm) * 100
        details["virtual_px"] = virtual_px
        details["virtual_cm"] = (virtual_px / ppm) * 100
        
        return DistanceEstimate(
            total_px=chosen_total,
            virtual_px=virtual_px,
            method=method,
            confidence=confidence,
            details=details
        )
    
    def _log_shot_validation(
        self,
        estimate: DistanceEstimate,
        trajectory_fit_status: str,
        exit_r_squared: float
    ):
        """Print per-shot validation for debugging."""
        ppm = self._current_pixels_per_meter
        d = estimate.details
        
        # Get raw 2-frame v0 for comparison (shows why we don't use it)
        v0_raw = 0.0
        if len(self._velocity_history) > 0:
            v0_raw = self._velocity_history[0].speed
        
        logger.info("=" * 70)
        logger.info("SHOT VALIDATION REPORT")
        logger.info("=" * 70)
        logger.info(f"  v0_raw (2-frame):       {v0_raw:.1f} px/s ({v0_raw/ppm:.3f} m/s) [NOT USED]")
        logger.info(f"  v0_robust (regression): {d.get('v0_robust_px_s', 0):.1f} px/s ({d.get('v0_robust_px_s', 0)/ppm:.3f} m/s)")
        if d.get('v0_robust_px_s', 0) > 0:
            logger.info(f"    R²={d.get('v0_robust_r_squared', 0):.3f}, "
                       f"frames={d.get('v0_robust_frames', 0)}, "
                       f"residual={d.get('v0_robust_residual', 0):.2f}px, "
                       f"trustworthy={d.get('v0_trustworthy', False)}")
        logger.info(f"  v_exit_weighted:        {d.get('exit_velocity_px_s', 0):.1f} px/s ({d.get('exit_velocity_px_s', 0)/ppm:.3f} m/s)")
        logger.info(f"    exit_r_squared={exit_r_squared:.3f}, conf={d.get('exit_confidence', 0):.2f}")
        logger.info("-" * 70)
        logger.info(f"  trajectory_fit:         {trajectory_fit_status}")
        logger.info(f"  d_total_v0:             {d.get('d_total_v0_cm', 0):.1f} cm")
        logger.info(f"  d_total_exit:           {d.get('d_total_exit_cm', 0):.1f} cm")
        logger.info("-" * 70)
        logger.info(f"  CHOSEN METHOD:          {estimate.method.upper()}")
        logger.info(f"  PHYSICAL:               {d.get('physical_cm', 0):.1f} cm")
        logger.info(f"  VIRTUAL:                {d.get('virtual_cm', 0):.1f} cm")
        logger.info(f"  TOTAL:                  {d.get('chosen_total_cm', 0):.1f} cm")
        logger.info("=" * 70)
    
    def _log_state_timeline_entry(self, timestamp_ns: int, frame_id: int, event: str, details: dict = None):
        """Add an entry to the state timeline for debugging."""
        entry = StateTimelineEntry(
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            state=self._state.value,
            event=event,
            details=details or {}
        )
        self._state_timeline.append(entry)
    
    def _fit_trajectory_physics(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Fit linear deceleration physics model to trajectory data.
        
        Uses the fundamental physics relationship: v² = v₀² - 2ad
        Where:
        - v = current velocity
        - v₀ = initial velocity
        - a = deceleration
        - d = distance traveled
        
        By fitting this to all trajectory points, we get:
        - More accurate initial velocity (uses all data, not just first frames)
        - Actual deceleration on YOUR surface (not assumed)
        - Predicted total distance = v₀²/(2a)
        
        Returns:
            (initial_speed_px_s, deceleration_px_s2, predicted_total_distance_px)
            or (None, None, None) if insufficient data or fit fails
        """
        if len(self._raw_trajectory) < 8:
            logger.debug(f"Trajectory fitting: insufficient data ({len(self._raw_trajectory)} < 8 points)")
            return None, None, None
        
        points = list(self._raw_trajectory)
        
        # Calculate instantaneous speeds and cumulative distances
        speeds = []
        distances = []
        
        cum_dist = 0.0
        for i in range(1, len(points)):
            dt = self._get_dt_from_timestamps(points[i-1].timestamp_ns, points[i].timestamp_ns)
            if dt <= 0:
                continue
            
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            step_dist = np.sqrt(dx*dx + dy*dy)
            speed = step_dist / dt
            
            cum_dist += step_dist
            
            # Skip outliers (sudden speed jumps often indicate detection noise)
            if len(speeds) > 0:
                prev_speed = speeds[-1]
                if abs(speed - prev_speed) > prev_speed * 0.5 and abs(speed - prev_speed) > 200:
                    logger.debug(f"Trajectory fitting: skipping outlier speed {speed:.1f} (prev={prev_speed:.1f})")
                    continue
            
            speeds.append(speed)
            distances.append(cum_dist)
        
        if len(speeds) < 5:
            logger.debug(f"Trajectory fitting: insufficient valid speeds ({len(speeds)} < 5)")
            return None, None, None
        
        # Fit v² = v₀² - 2ad using least squares
        # Rewrite as: v² = b + m*d where b = v₀², m = -2a
        # This is linear regression: y = b + m*x
        
        v_squared = [s*s for s in speeds]
        
        n = len(v_squared)
        sum_x = sum(distances)
        sum_y = sum(v_squared)
        sum_xy = sum(d*vs for d, vs in zip(distances, v_squared))
        sum_xx = sum(d*d for d in distances)
        
        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-10:
            logger.debug("Trajectory fitting: degenerate data (zero denominator)")
            return None, None, None
        
        slope = (n * sum_xy - sum_x * sum_y) / denom  # This is -2a
        intercept = (sum_y - slope * sum_x) / n  # This is v₀²
        
        # Validate fit results
        if intercept <= 0:
            logger.debug(f"Trajectory fitting: invalid intercept {intercept:.1f} (v₀² must be positive)")
            return None, None, None
        
        if slope >= 0:
            # Ball accelerating or no deceleration - physics doesn't make sense
            logger.debug(f"Trajectory fitting: invalid slope {slope:.1f} (should be negative for deceleration)")
            return None, None, None
        
        v0 = np.sqrt(intercept)
        a = -slope / 2
        
        # Predicted total distance = v₀² / (2a)
        total_dist = intercept / (-slope)
        
        # Calculate R² (coefficient of determination) to assess fit quality
        y_mean = sum_y / n
        ss_tot = sum((vs - y_mean)**2 for vs in v_squared)
        ss_res = sum((vs - (intercept + slope * d))**2 for vs, d in zip(v_squared, distances))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        a_m_s2 = a / self._current_pixels_per_meter
        
        logger.info(f"Trajectory physics fit: v₀={v0:.1f}px/s, a={a:.1f}px/s² ({a_m_s2:.3f}m/s²), "
                   f"predicted_dist={total_dist:.1f}px, R²={r_squared:.3f}, points={n}")
        
        # GUARDRAILS: Reject unrealistic fitted values
        # These indicate tracking issues (motion blur, lost frames) rather than true physics
        
        # 1. Deceleration sanity check: realistic putting surfaces are 0.25-0.9 m/s²
        if a_m_s2 < 0.25 or a_m_s2 > 0.9:
            logger.warning(f"Trajectory fitting: deceleration {a_m_s2:.2f} m/s² outside realistic range [0.25, 0.9] - rejecting fit")
            return None, None, None
        
        # 2. R² quality check: low R² means noisy/unreliable fit
        # Lowered threshold from 0.85 to 0.70 - trajectory fit often better than 2-frame measurement
        if r_squared < 0.70:
            logger.warning(f"Trajectory fitting: R²={r_squared:.3f} too low (need >0.70) - rejecting fit")
            return None, None, None
        
        # 3. Minimum data points for reliable fit (lowered from 15 to 8)
        if n < 8:
            logger.warning(f"Trajectory fitting: only {n} points (need ≥8) - rejecting fit")
            return None, None, None
        
        logger.info(f"Trajectory fit ACCEPTED: decel={a_m_s2:.3f}m/s², R²={r_squared:.3f}")
        return v0, a, total_dist
    
    def _check_transitions(self, detection: Detection, timestamp_ns: int, frame_id: int):
        """Check for state machine transitions."""
        if self._state == ShotState.ARMED:
            self._check_armed_to_tracking(detection, frame_id)
        elif self._state == ShotState.TRACKING:
            self._check_tracking_to_stopped(frame_id, timestamp_ns)
        elif self._state == ShotState.STOPPED:
            self._transition_to_cooldown(timestamp_ns)
    
    def _check_armed_to_tracking(self, detection: Detection, frame_id: int):
        """
        Check if motion threshold exceeded to start tracking.
        
        Uses displacement from locked position.
        Requires MOTION_CONFIRM_FRAMES consecutive triggers to prevent false shots.
        Must maintain motion direction (not just jitter in place).
        Motion direction must be within VALID_MOTION_ANGLE_DEG of forward direction.
        Ball signature must be validated (not putter/hand).
        """
        if self._smoothed_pos is None:
            return
        
        # Skip motion detection during settling period (ball just placed)
        if self._settling_countdown > 0:
            self._settling_countdown -= 1
            if self._settling_countdown == 0:
                logger.info("Settling period complete, motion detection enabled")
            return
        
        new_pos = (detection.cx, detection.cy)
        
        # Use locked position if available for more stable trigger
        ref_pos = self._locked_pos if self._locked_pos else self._smoothed_pos
        
        # Calculate displacement and direction
        dx = new_pos[0] - ref_pos[0]
        dy = new_pos[1] - ref_pos[1]
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Motion triggered only by significant displacement
        motion_detected = displacement > self.MOTION_THRESHOLD_PX
        
        if motion_detected:
            # Validate ball signature first (must look like a ball, not putter)
            if not self._validate_ball_signature(detection):
                logger.debug(f"Motion detected but ball signature invalid - waiting for stable ball")
                return
            
            # Check motion direction against valid putting direction
            motion_direction = np.degrees(np.arctan2(dy, dx))
            
            # Calculate angle difference from forward direction
            angle_diff = abs(motion_direction - self._forward_direction_deg)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Reject motion that's not within valid angle of forward direction
            if angle_diff > self._valid_motion_angle_deg:
                logger.debug(
                    f"Motion rejected: direction {motion_direction:.1f}° not within "
                    f"+/-{self._valid_motion_angle_deg:.1f}° of forward {self._forward_direction_deg:.1f}° "
                    f"(diff={angle_diff:.1f}°)"
                )
                self._motion_trigger_count = 0
                if hasattr(self, '_motion_start_pos'):
                    del self._motion_start_pos
                return
            
            self._motion_trigger_count += 1
            
            # Track accumulated motion to distinguish real motion from jitter
            if not hasattr(self, '_motion_start_pos'):
                self._motion_start_pos = ref_pos
            
            # Check if motion is sustained (ball actually moved in a direction)
            accumulated_dist = np.sqrt(
                (new_pos[0] - self._motion_start_pos[0])**2 +
                (new_pos[1] - self._motion_start_pos[1])**2
            )
            
            logger.debug(
                f"Motion trigger {self._motion_trigger_count}/{self.MOTION_CONFIRM_FRAMES}: "
                f"displacement={displacement:.1f}px, accumulated={accumulated_dist:.1f}px, "
                f"direction={motion_direction:.1f}° (valid)"
            )
            
            # Require both consecutive triggers AND accumulated distance
            if self._motion_trigger_count >= self.MOTION_CONFIRM_FRAMES and accumulated_dist > self.MOTION_THRESHOLD_PX * 1.5:
                self._transition_to_tracking(frame_id)
        else:
            self._motion_trigger_count = 0
            if hasattr(self, '_motion_start_pos'):
                del self._motion_start_pos
    
    def _transition_to_tracking(self, frame_id: int):
        """Transition from ARMED to TRACKING."""
        logger.info(f"ARMED -> TRACKING at frame {frame_id}")
        
        # Store the shot start position BEFORE clearing anything
        # Use _motion_start_pos if available - this is where the ball ACTUALLY was
        # when motion first began, before the confirmation delay
        if hasattr(self, '_motion_start_pos') and self._motion_start_pos is not None:
            self._shot_start_pos = self._motion_start_pos
            logger.info(f"Using motion_start_pos as shot origin: {self._shot_start_pos}")
        else:
            # Fallback to current position
            self._shot_start_pos = self._current_pos
            logger.info(f"Using current_pos as shot origin (fallback): {self._shot_start_pos}")
        
        self._state = ShotState.TRACKING
        self._lane = TrackerLane.MOTION
        self._motion_start_frame = frame_id
        self._first_speed_frame = 0
        self._shot_result = None
        self._motion_trigger_count = 0
        self._stopped_count = 0
        
        # Clear BOTH trajectories for clean tracking of this shot
        self._trajectory.clear()
        self._raw_trajectory.clear()
        
        # Clear shot timestamps for timing stats
        self._shot_timestamps = []
        
        # Add the start position as first point of the new trajectory
        if self._shot_start_pos:
            self._trajectory.append(TrackPoint(
                x=self._shot_start_pos[0],
                y=self._shot_start_pos[1],
                timestamp_ns=0,  # Will be updated on next frame
                frame_id=frame_id,
                confidence=1.0
            ))
        
        # Clear locked position - no longer needed in motion
        self._locked_pos = None
        self._idle_stability_count = 0
        
        # Clear motion start tracking
        if hasattr(self, '_motion_start_pos'):
            del self._motion_start_pos
        
        # Reset ball validation
        self._ball_stable_count = 0
    
    def _check_tracking_to_stopped(self, frame_id: int, timestamp_ns: int):
        """Check if ball has stopped moving using timestamp-based detection."""
        # Don't transition to STOPPED if we recently lost detection but had high velocity
        # This indicates ball might have exited frame, not stopped
        if self._frames_lost > 0 and self._last_valid_velocity:
            if self._last_valid_velocity.speed > self.MIN_EXIT_SPEED_PX_S:
                # Ball was moving fast when we lost it - don't call it "stopped"
                # Frame exit detection should handle this
                logger.debug(f"Ignoring stop check: frames_lost={self._frames_lost}, "
                           f"last_speed={self._last_valid_velocity.speed:.1f}")
                return
        
        is_low_velocity = (self._velocity is None or 
                          self._velocity.speed < self.STOPPED_VELOCITY_THRESHOLD)
        
        if is_low_velocity:
            # Start or continue low velocity tracking
            if self._low_velocity_start_ns is None:
                self._low_velocity_start_ns = timestamp_ns
                logger.debug(f"Low velocity detected, starting stop timer")
            else:
                # Check if we've been at low velocity for the required duration
                duration_ms = (timestamp_ns - self._low_velocity_start_ns) / 1e6
                if duration_ms >= self._stopped_confirm_time_ms:
                    logger.info(f"Ball stopped: low velocity for {duration_ms:.1f}ms >= {self._stopped_confirm_time_ms}ms")
                    self._transition_to_stopped(frame_id)
        else:
            # Reset low velocity tracking
            if self._low_velocity_start_ns is not None:
                logger.debug(f"Velocity above threshold, resetting stop timer")
            self._low_velocity_start_ns = None
    
    def _transition_to_stopped(self, frame_id: int):
        """Transition from TRACKING to STOPPED."""
        logger.info(f"TRACKING -> STOPPED at frame {frame_id}")
        
        # Check if this is a valid shot (minimum distance traveled)
        if len(self._trajectory) >= 2:
            start = self._trajectory[0]
            end = self._trajectory[-1]
            physical_dist = np.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
            
            if physical_dist < self.MIN_SHOT_DISTANCE_PX:
                # Too short - likely a false trigger (putter swing, hand movement)
                logger.warning(f"Shot rejected: physical distance {physical_dist:.1f}px < minimum {self.MIN_SHOT_DISTANCE_PX}px (likely false trigger)")
                self._transition_to_armed()
                return
        
        self._state = ShotState.STOPPED
        
        # Compute shot timing stats
        self._compute_shot_timing_stats()
        
        # Compute shot result
        self._compute_shot_result(frame_id)
    
    def _compute_shot_timing_stats(self):
        """Compute frame timing statistics for the shot."""
        if len(self._shot_timestamps) < 2:
            self._shot_timing_stats = None
            return
        
        # Calculate dt between consecutive frames
        dts = []
        for i in range(1, len(self._shot_timestamps)):
            dt_ms = (self._shot_timestamps[i] - self._shot_timestamps[i-1]) / 1e6
            if dt_ms > 0:
                dts.append(dt_ms)
        
        if not dts:
            self._shot_timing_stats = None
            return
        
        dts = np.array(dts)
        effective_fps = 1000.0 / np.mean(dts) if np.mean(dts) > 0 else 0
        
        self._shot_timing_stats = FrameTimingStats(
            effective_fps=effective_fps,
            dt_mean_ms=float(np.mean(dts)),
            dt_std_ms=float(np.std(dts)),
            dt_min_ms=float(np.min(dts)),
            dt_max_ms=float(np.max(dts)),
            frame_count=len(self._shot_timestamps)
        )
        
        logger.info(f"Shot timing: fps={effective_fps:.1f}, dt={np.mean(dts):.2f}±{np.std(dts):.2f}ms, "
                   f"frames={len(self._shot_timestamps)}")
    
    def _compute_shot_result(self, frame_id: int):
        """Compute final shot metrics. Called once when shot stops - values are frozen."""
        trajectory_points = [(p.x, p.y) for p in self._trajectory]
        
        # Get initial velocity (use stored impact velocity or compute)
        initial_speed = 0.0
        initial_direction = 0.0
        
        if self._impact_velocity:
            initial_speed = self._impact_velocity.speed
            initial_direction = self._impact_velocity.direction_deg
        elif self._velocity_history:
            # Use first valid velocity
            for v in self._velocity_history:
                if v.speed > self.STOPPED_VELOCITY_THRESHOLD:
                    initial_speed = v.speed
                    initial_direction = v.direction_deg
                    break
        
        # Compute timing metrics
        frames_to_tracking = self.MOTION_CONFIRM_FRAMES
        frames_to_speed = self._first_speed_frame - self._motion_start_frame if self._first_speed_frame > 0 else 0
        
        # Duration (including virtual rolling time)
        if len(self._trajectory) >= 2:
            start_t = self._trajectory[0].timestamp_ns
            end_t = self._trajectory[-1].timestamp_ns
            duration_ms = (end_t - start_t) / 1e6
        else:
            duration_ms = 0.0
        
        # Add virtual rolling time if applicable
        if self._virtual_ball:
            duration_ms += self._virtual_ball.time_since_exit * 1000
        
        # Physical distance (trajectory in frame) - FROZEN at computation time
        physical_distance_px = 0.0
        if trajectory_points:
            start_pos = trajectory_points[0]
            end_pos = trajectory_points[-1]
            physical_distance_px = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Virtual distance (after frame exit) - FROZEN at computation time
        virtual_distance_px = 0.0
        exited_frame = False
        if self._virtual_ball:
            virtual_distance_px = self._virtual_ball.distance_traveled
            exited_frame = True
        
        # Total distance - FROZEN at computation time
        total_distance_px = physical_distance_px + virtual_distance_px
        
        # Freeze the calibration value used for this shot
        frozen_ppm = self._current_pixels_per_meter
        
        # Log trajectory info for debugging
        logger.info(f"Shot result FROZEN: {len(trajectory_points)} points, "
                   f"physical={physical_distance_px:.1f}px, virtual={virtual_distance_px:.1f}px, "
                   f"total={total_distance_px:.1f}px, ppm={frozen_ppm:.1f}")
        
        self._shot_result = ShotResult(
            initial_speed_px_s=initial_speed,
            initial_direction_deg=initial_direction,
            frames_to_tracking=frames_to_tracking,
            frames_to_speed=frames_to_speed,
            trajectory=trajectory_points,
            duration_ms=duration_ms,
            physical_distance_px=physical_distance_px,
            virtual_distance_px=virtual_distance_px,
            total_distance_px=total_distance_px,
            exited_frame=exited_frame,
            pixels_per_meter=frozen_ppm
        )
        
        logger.info(f"Shot result: speed={initial_speed:.1f}px/s, dir={initial_direction:.1f}°, "
                   f"total_distance={total_distance_px:.1f}px, exited_frame={exited_frame}")
    
    def _transition_to_cooldown(self, timestamp_ns: int):
        """Transition from STOPPED to COOLDOWN."""
        logger.info("STOPPED -> COOLDOWN")
        self._state = ShotState.COOLDOWN
        self._cooldown_start_ns = timestamp_ns
    
    def _handle_cooldown(self, timestamp_ns: int):
        """Handle cooldown state, transition to ARMED when done."""
        elapsed_ms = (timestamp_ns - self._cooldown_start_ns) / 1e6
        if elapsed_ms >= self.COOLDOWN_DURATION_MS:
            self._transition_to_armed()
    
    def _transition_to_armed(self):
        """Transition from COOLDOWN to ARMED."""
        logger.info("COOLDOWN -> ARMED")
        self._state = ShotState.ARMED
        self._lane = TrackerLane.IDLE
        self._trajectory.clear()
        self._raw_trajectory.clear()
        self._velocity_history.clear()
        self._velocity = None
        self._motion_trigger_count = 0
        self._stopped_count = 0
        self._idle_positions.clear()
        self._locked_pos = self._current_pos  # Lock at current position
        self._idle_stability_count = 0
        self._settling_countdown = self.SETTLING_FRAMES // 2  # Brief settling after shot
        self._roi = None
        self._background.reset()  # Reset background for new baseline
        
        # Clear virtual ball state
        self._virtual_ball = None
        self._exit_state = None
        self._frames_lost = 0
        self._last_valid_position = None
        self._last_valid_velocity = None
        self._fitted_deceleration = None
        self._fitted_physics = None
        self._expected_total_distance_px = 0.0
        self._expected_virtual_distance_px = 0.0
        
        # Clear ball validation
        self._ball_stable_count = 0
        self._last_ball_check_radius = 0.0
        
        # Clear shot timing
        self._shot_timestamps = []
        self._shot_timing_stats = None
        
        # Clear new determinism-related state
        self._state_timeline = []
        self._robust_v0 = None
        self._low_velocity_start_ns = None
        self._lost_track_count = 0
    
    def _compute_idle_stddev(self) -> float:
        """Compute position standard deviation when idle."""
        if len(self._idle_positions) < 10:
            return 0.0
        
        positions = np.array(list(self._idle_positions))
        return float(np.std(positions, axis=0).mean())
    
    def _build_state(self) -> TrackerState:
        """Build current state for external consumption."""
        # During virtual rolling, use virtual ball position
        if self._state == ShotState.VIRTUAL_ROLLING and self._virtual_ball:
            ball_x = self._virtual_ball.x
            ball_y = self._virtual_ball.y
        else:
            ball_x = self._current_pos[0] if self._current_pos else None
            ball_y = self._current_pos[1] if self._current_pos else None
        
        return TrackerState(
            state=self._state,
            lane=self._lane,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_radius=self._last_radius,
            ball_confidence=self._last_confidence,
            velocity=self._velocity,
            shot_result=self._shot_result,
            idle_stddev=self._compute_idle_stddev(),
            virtual_ball=self._virtual_ball,
            exit_state=self._exit_state
        )
    
    def reset(self):
        """Reset tracker to initial state."""
        self._state = ShotState.ARMED
        self._lane = TrackerLane.IDLE
        self._current_pos = None
        self._smoothed_pos = None
        self._locked_pos = None
        self._last_radius = None
        self._trajectory.clear()
        self._raw_trajectory.clear()
        self._idle_positions.clear()
        self._velocity_history.clear()
        self._velocity = None
        self._shot_result = None
        self._shot_start_pos = None
        self._background.reset()
        self._roi = None
        self._foreground_delta = 0.0
        self._motion_trigger_count = 0
        self._idle_stability_count = 0
        self._settling_countdown = self.SETTLING_FRAMES  # Start with settling period
        
        # Clear virtual ball state
        self._virtual_ball = None
        self._exit_state = None
        self._frames_lost = 0
        self._last_valid_position = None
        self._last_valid_velocity = None
        self._fitted_deceleration = None
        self._fitted_physics = None
        self._expected_total_distance_px = 0.0
        self._expected_virtual_distance_px = 0.0
        
        # Clear ball validation
        self._ball_stable_count = 0
        self._last_ball_check_radius = 0.0
        
        # Clear timing
        self._shot_timestamps = []
        self._shot_timing_stats = None
        
        # Clear new determinism-related state
        self._state_timeline = []
        self._robust_v0 = None
        self._low_velocity_start_ns = None
        self._lost_track_count = 0
        
        logger.info("Tracker reset")
    
    def set_calibration(self, pixels_per_meter: float):
        """
        Update the current calibration for pixel/meter conversions.
        Called by main loop when auto-calibration updates.
        """
        self._current_pixels_per_meter = pixels_per_meter
    
    def get_deceleration_px_s2(self) -> float:
        """
        Get deceleration in px/s² using current calibration.
        """
        return self.DECELERATION_M_S2 * self._current_pixels_per_meter
    
    def set_deceleration_m_s2(self, deceleration: float):
        """Set deceleration in m/s²."""
        self.DECELERATION_M_S2 = deceleration
        logger.info(f"Virtual ball deceleration set to {deceleration:.3f} m/s²")
    
    def set_forward_direction(self, direction_deg: float):
        """
        Set the forward putting direction (from calibration).
        Motion must be within VALID_MOTION_ANGLE_DEG of this direction to trigger tracking.
        """
        self._forward_direction_deg = direction_deg
        logger.info(f"Forward direction set to {direction_deg:.1f}°")
    
    def set_valid_motion_angle(self, angle_deg: float):
        """Set the valid motion angle tolerance (+/- from forward direction)."""
        self._valid_motion_angle_deg = angle_deg
        logger.info(f"Valid motion angle set to +/-{angle_deg:.1f}°")
    
    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        """Get current trajectory as list of (x, y) tuples."""
        return [(p.x, p.y) for p in self._trajectory]
    
    @property
    def effective_fps(self) -> float:
        """Get effective FPS from actual frame timestamps."""
        return self._effective_fps
    
    @property
    def shot_timing_stats(self) -> Optional[FrameTimingStats]:
        """Get timing statistics from the last shot."""
        return self._shot_timing_stats
    
    def set_frame_bounds(self, width: int, height: int):
        """Set frame dimensions for exit detection."""
        self._frame_width = width
        self._frame_height = height
        logger.info(f"Frame bounds set to {width}x{height}")
    
    def _compute_trajectory_curvature(self) -> float:
        """
        Compute curvature from trajectory points.
        
        Uses the last N points to fit a curve and estimate the rate of direction change.
        Positive = curving right, Negative = curving left.
        
        Returns curvature in degrees per pixel traveled.
        """
        if len(self._raw_trajectory) < 10:
            return 0.0
        
        points = list(self._raw_trajectory)[-20:]  # Use last 20 points
        if len(points) < 10:
            return 0.0
        
        # Compute direction at start and end of the window
        # Start direction (first few points)
        p0, p1 = points[0], points[min(5, len(points)-1)]
        dx1 = p1.x - p0.x
        dy1 = p1.y - p0.y
        dist1 = np.sqrt(dx1**2 + dy1**2)
        if dist1 < 5:
            return 0.0
        angle1 = np.degrees(np.arctan2(dy1, dx1))
        
        # End direction (last few points)
        p2, p3 = points[-min(6, len(points))], points[-1]
        dx2 = p3.x - p2.x
        dy2 = p3.y - p2.y
        dist2 = np.sqrt(dx2**2 + dy2**2)
        if dist2 < 5:
            return 0.0
        angle2 = np.degrees(np.arctan2(dy2, dx2))
        
        # Angle change
        angle_diff = angle2 - angle1
        # Normalize to -180 to 180
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        # Total distance traveled in this window
        total_dist = 0.0
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            total_dist += np.sqrt(dx**2 + dy**2)
        
        if total_dist < 10:
            return 0.0
        
        # Curvature = angle change per unit distance
        curvature = angle_diff / total_dist
        
        return curvature
    
    def _check_frame_exit(self, detection: Optional[Detection], timestamp_ns: int, frame_id: int) -> bool:
        """
        Check if ball has exited the frame during tracking.
        
        Handles exit from ANY edge based on ball velocity direction.
        Returns True if ball has exited and virtual rolling should begin.
        """
        if self._state != ShotState.TRACKING:
            return False
        
        # Track frames without detection
        if detection is None:
            self._frames_lost += 1
            logger.debug(f"Frame {frame_id}: detection lost, frames_lost={self._frames_lost}, "
                        f"last_pos={self._last_valid_position}, "
                        f"last_vel={self._last_valid_velocity.speed if self._last_valid_velocity else None}")
        else:
            self._frames_lost = 0
            self._last_valid_position = (detection.cx, detection.cy)
            if self._velocity:
                self._last_valid_velocity = self._velocity
        
        should_exit = False
        exit_reason = ""
        
        # METHOD 1: Ball lost AND had significant velocity
        # If we were tracking a fast-moving ball and suddenly lost it, it likely exited
        if self._frames_lost >= 2 and self._last_valid_velocity:
            speed = self._last_valid_velocity.speed
            if speed > self.MIN_EXIT_SPEED_PX_S:
                # Ball was moving fast and we lost it - assume it exited
                should_exit = True
                exit_reason = f"Lost fast-moving ball (speed={speed:.1f}px/s)"
        
        # METHOD 2: Ball is near edge AND heading toward that edge
        if self._last_valid_position and self._last_valid_velocity:
            x, y = self._last_valid_position
            vx, vy = self._last_valid_velocity.vx, self._last_valid_velocity.vy
            speed = self._last_valid_velocity.speed
            
            # Calculate time to edge based on velocity
            margin = self.FRAME_EXIT_MARGIN_PX * 3  # Larger margin for prediction
            
            # Check if ball would reach any edge soon based on trajectory
            if speed > self.MIN_EXIT_SPEED_PX_S:
                # Right edge
                if vx > 50 and x >= self._frame_width - margin:
                    if self._frames_lost >= 1:
                        should_exit = True
                        exit_reason = f"Ball exited RIGHT edge (x={x:.1f}, vx={vx:.1f})"
                
                # Left edge  
                if vx < -50 and x <= margin:
                    if self._frames_lost >= 1:
                        should_exit = True
                        exit_reason = f"Ball exited LEFT edge (x={x:.1f}, vx={vx:.1f})"
                
                # Bottom edge
                if vy > 50 and y >= self._frame_height - margin:
                    if self._frames_lost >= 1:
                        should_exit = True
                        exit_reason = f"Ball exited BOTTOM edge (y={y:.1f}, vy={vy:.1f})"
                
                # Top edge
                if vy < -50 and y <= margin:
                    if self._frames_lost >= 1:
                        should_exit = True
                        exit_reason = f"Ball exited TOP edge (y={y:.1f}, vy={vy:.1f})"
        
        # METHOD 3: Ball is literally at edge with good velocity
        if detection and self._velocity and self._velocity.speed > self.MIN_EXIT_SPEED_PX_S:
            x, y = detection.cx, detection.cy
            margin = self.FRAME_EXIT_MARGIN_PX
            
            at_edge = (x <= margin or x >= self._frame_width - margin or
                      y <= margin or y >= self._frame_height - margin)
            
            if at_edge:
                should_exit = True
                exit_reason = f"Ball at edge with velocity: pos=({x:.1f}, {y:.1f}), speed={self._velocity.speed:.1f}"
        
        if should_exit:
            logger.info(f"Frame exit detected: {exit_reason}")
            
            # Check minimum requirements for virtual rolling
            if len(self._raw_trajectory) < self.MIN_TRACKING_FRAMES_FOR_EXIT:
                logger.warning(f"Not enough tracking frames for virtual rolling: {len(self._raw_trajectory)} < {self.MIN_TRACKING_FRAMES_FOR_EXIT}")
                return False
            
            vel = self._last_valid_velocity or self._velocity
            if vel is None or vel.speed < self.MIN_EXIT_SPEED_PX_S:
                logger.warning(f"Exit speed too low for virtual rolling: {vel.speed if vel else 0:.1f} px/s < {self.MIN_EXIT_SPEED_PX_S}")
                return False
            
            logger.info(f"=== VIRTUAL ROLLING TRIGGERED ===")
            logger.info(f"  Speed: {vel.speed:.1f} px/s")
            logger.info(f"  Direction: {vel.direction_deg:.1f}°")
            logger.info(f"  Trajectory points: {len(self._raw_trajectory)}")
            return True
        
        return False
    
    def _transition_to_virtual_rolling(self, timestamp_ns: int, frame_id: int):
        """
        Transition from TRACKING to VIRTUAL_ROLLING when ball exits frame.
        
        Uses the new deterministic distance estimation strategy:
        1. Trajectory physics fit - if passes guardrails (best)
        2. Exit velocity - PRIMARY fallback when fit rejected
        3. Robust v0 regression - SECONDARY fallback only if exit confidence low
        
        NEVER uses 2-frame v0 for distance calculation.
        """
        logger.info(f"TRACKING -> VIRTUAL_ROLLING at frame {frame_id}")
        
        # Log state timeline entry
        self._log_state_timeline_entry(timestamp_ns, frame_id, "transition_to_virtual_rolling")
        
        # Capture exit state
        exit_pos = self._last_valid_position or self._current_pos
        
        # Get exit velocity via regression (preferred method)
        exit_vel_regression, exit_r_squared = self._compute_exit_velocity_regression()
        exit_vel = exit_vel_regression
        
        # Fallback to weighted average if regression failed
        if exit_vel is None:
            exit_vel = self._compute_exit_velocity()
            exit_r_squared = 0.5  # Lower confidence for weighted average
        
        if exit_vel is None:
            exit_vel = self._last_valid_velocity or self._velocity
            exit_r_squared = 0.3  # Even lower confidence
        
        if exit_pos is None or exit_vel is None:
            logger.error("Cannot transition to virtual rolling: missing position/velocity")
            self._transition_to_stopped(frame_id)
            return
        
        logger.info(f"Exit velocity: {exit_vel.speed:.1f} px/s @ {exit_vel.direction_deg:.1f}° (R²={exit_r_squared:.3f})")
        
        # Compute robust v0 using regression on frames [motion_start+2..motion_start+10]
        robust_v0 = self._compute_robust_v0()
        self._robust_v0 = robust_v0  # Store for later use
        
        curvature = self._compute_trajectory_curvature()
        
        # Calculate physical distance traveled so far
        physical_distance_px = 0.0
        if self._shot_start_pos and exit_pos:
            physical_distance_px = np.sqrt(
                (exit_pos[0] - self._shot_start_pos[0])**2 + 
                (exit_pos[1] - self._shot_start_pos[1])**2
            )
        
        # Try to fit physics model to trajectory
        fitted_v0, fitted_a, fitted_total_dist = self._fit_trajectory_physics()
        
        # Build trajectory fit result for distance estimation
        trajectory_fit_result = None
        trajectory_fit_status = "REJECTED"
        if fitted_v0 is not None and fitted_a is not None and fitted_total_dist is not None:
            fitted_a_m_s2 = fitted_a / self._current_pixels_per_meter
            # Use fitted values if deceleration is within reasonable range [0.30, 0.80] m/s²
            if 0.30 < fitted_a_m_s2 < 0.80:
                trajectory_fit_result = (fitted_v0, fitted_a, fitted_total_dist)
                trajectory_fit_status = f"ACCEPTED (a={fitted_a_m_s2:.3f}m/s²)"
            else:
                trajectory_fit_status = f"REJECTED (a={fitted_a_m_s2:.3f}m/s² outside [0.30, 0.80])"
        else:
            trajectory_fit_status = "REJECTED (insufficient data or fit failed)"
        
        # Use the new deterministic distance estimation
        distance_estimate = self._compute_distance_estimate(
            physical_distance_px=physical_distance_px,
            trajectory_fit_result=trajectory_fit_result,
            exit_velocity=exit_vel,
            exit_r_squared=exit_r_squared,
            robust_v0=robust_v0
        )
        
        # Log the shot validation report
        self._log_shot_validation(distance_estimate, trajectory_fit_status, exit_r_squared)
        
        expected_total_distance_px = distance_estimate.total_px
        expected_virtual_distance_px = distance_estimate.virtual_px
        
        # Get configured deceleration
        a = self.get_deceleration_px_s2()
        a_m_s2 = a / self._current_pixels_per_meter
        
        # Determine exit speed for virtual ball animation
        # Use exit velocity for animation (consistent with direction)
        robust_exit_speed = exit_vel.speed
        
        # Scale velocity components to match robust exit speed
        if exit_vel.speed > 0:
            virtual_vx = exit_vel.vx
            virtual_vy = exit_vel.vy
        else:
            # Fallback: use last known direction
            direction_rad = np.radians(exit_vel.direction_deg)
            virtual_vx = robust_exit_speed * np.cos(direction_rad)
            virtual_vy = robust_exit_speed * np.sin(direction_rad)
        
        self._exit_state = ExitState(
            position=exit_pos,
            velocity=(virtual_vx, virtual_vy),
            speed=robust_exit_speed,
            direction_deg=exit_vel.direction_deg,
            curvature=curvature,
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            trajectory_before_exit=[(p.x, p.y) for p in self._trajectory]
        )
        
        # Store for later use
        self._fitted_physics = (fitted_v0, fitted_a, fitted_total_dist) if fitted_a else None
        self._expected_total_distance_px = expected_total_distance_px
        self._expected_virtual_distance_px = expected_virtual_distance_px
        
        # Calculate timing
        expected_stop_time = robust_exit_speed / a if robust_exit_speed > 0 else 0
        
        # Initialize virtual ball at exit position
        self._virtual_ball = VirtualBallState(
            x=exit_pos[0],
            y=exit_pos[1],
            vx=virtual_vx,
            vy=virtual_vy,
            speed=robust_exit_speed,
            distance_traveled=0.0,
            time_since_exit=0.0,
            is_rolling=True,
            final_position=None
        )
        
        self._virtual_start_time_ns = timestamp_ns
        self._state = ShotState.VIRTUAL_ROLLING
        
        logger.info(f"=== VIRTUAL ROLLING STARTED ===")
        logger.info(f"  Exit position: {exit_pos}")
        logger.info(f"  Physical distance: {physical_distance_px:.1f} px ({physical_distance_px/self._current_pixels_per_meter*100:.1f} cm)")
        logger.info(f"  Exit speed: {robust_exit_speed:.1f} px/s")
        logger.info(f"  Exit direction: {exit_vel.direction_deg:.1f}°")
        logger.info(f"  Curvature: {curvature:.4f}°/px")
        logger.info(f"  Deceleration: {a:.1f} px/s² ({a_m_s2:.3f} m/s²)")
        logger.info(f"  Expected TOTAL distance: {expected_total_distance_px:.1f} px ({expected_total_distance_px/self._current_pixels_per_meter*100:.1f} cm)")
        logger.info(f"  Expected VIRTUAL distance: {expected_virtual_distance_px:.1f} px ({expected_virtual_distance_px/self._current_pixels_per_meter*100:.1f} cm)")
        logger.info(f"  Distance method: {distance_estimate.method.upper()}")
        logger.info(f"  Expected stop time: {expected_stop_time:.2f} s")
    
    def _update_virtual_ball(self, timestamp_ns: int, frame_id: int):
        """
        Update virtual ball position using physics simulation.
        
        Uses LINEAR deceleration (constant friction) for realistic putting feel.
        v(t) = v0 - a*t
        x(t) = v0*t - 0.5*a*t²
        """
        if self._virtual_ball is None or self._exit_state is None:
            logger.warning("_update_virtual_ball called but no virtual ball state!")
            return
        
        # Time since virtual rolling started
        dt = (timestamp_ns - self._virtual_start_time_ns) / 1e9
        
        if dt > self.MAX_VIRTUAL_TIME_S:
            logger.info(f"Virtual rolling timeout: dt={dt:.1f}s > max={self.MAX_VIRTUAL_TIME_S}s")
            self._finish_virtual_rolling(frame_id)
            return
        
        # LINEAR DECELERATION PHYSICS
        # v(t) = v0 - a*t
        # x(t) = v0*t - 0.5*a*t²
        # Ball stops at t_stop = v0/a
        
        initial_speed = self._exit_state.speed
        
        # Use configured deceleration (fitted deceleration already incorporated into exit speed)
        a = self.get_deceleration_px_s2()
        
        # Time when ball stops
        t_stop = initial_speed / a if a > 0 else 0
        
        # Check if ball has stopped
        if dt >= t_stop:
            # Ball has stopped - use final values
            current_speed = 0.0
            distance = (initial_speed ** 2) / (2 * a) if a > 0 else 0
            logger.info(f"Virtual ball stopped: t={dt:.2f}s >= t_stop={t_stop:.2f}s")
            self._finish_virtual_rolling(frame_id)
            return
        
        # Current speed (linear decrease)
        current_speed = initial_speed - a * dt
        
        if current_speed < self.MIN_VIRTUAL_SPEED_PX_S:
            logger.info(f"Virtual ball stopped: speed={current_speed:.1f} < threshold={self.MIN_VIRTUAL_SPEED_PX_S}")
            self._finish_virtual_rolling(frame_id)
            return
        
        # Distance traveled so far: x = v0*t - 0.5*a*t²
        distance = initial_speed * dt - 0.5 * a * dt * dt
        
        # Apply curvature to direction
        base_direction = np.radians(self._exit_state.direction_deg)
        curvature_rad = np.radians(self._exit_state.curvature)
        
        # Current direction = base direction + curvature * distance_traveled
        current_direction = base_direction + curvature_rad * distance
        
        # Position
        exit_x, exit_y = self._exit_state.position
        
        # For curved path, use average direction for position calculation
        if abs(self._exit_state.curvature) > 0.001:
            avg_direction = base_direction + curvature_rad * distance / 2
            x = exit_x + distance * np.cos(avg_direction)
            y = exit_y + distance * np.sin(avg_direction)
        else:
            x = exit_x + distance * np.cos(base_direction)
            y = exit_y + distance * np.sin(base_direction)
        
        # Update virtual ball state
        self._virtual_ball.x = x
        self._virtual_ball.y = y
        self._virtual_ball.vx = current_speed * np.cos(current_direction)
        self._virtual_ball.vy = current_speed * np.sin(current_direction)
        self._virtual_ball.speed = current_speed
        self._virtual_ball.distance_traveled = distance
        self._virtual_ball.time_since_exit = dt
        
        # Calculate final position (where ball will stop)
        max_distance = (initial_speed ** 2) / (2 * a) if a > 0 else 0
        if abs(self._exit_state.curvature) > 0.001:
            final_avg_dir = base_direction + curvature_rad * max_distance / 2
            final_x = exit_x + max_distance * np.cos(final_avg_dir)
            final_y = exit_y + max_distance * np.sin(final_avg_dir)
        else:
            final_x = exit_x + max_distance * np.cos(base_direction)
            final_y = exit_y + max_distance * np.sin(base_direction)
        
        self._virtual_ball.final_position = (final_x, final_y)
        
        # Log progress every ~0.5 seconds
        if int(dt * 2) != int((dt - 0.05) * 2):
            logger.info(f"Virtual rolling: t={dt:.2f}s/{t_stop:.2f}s, dist={distance:.1f}px, "
                       f"speed={current_speed:.1f}px/s, pos=({x:.1f}, {y:.1f})")
    
    def _finish_virtual_rolling(self, frame_id: int):
        """Finish virtual rolling and transition to STOPPED."""
        if self._virtual_ball:
            self._virtual_ball.is_rolling = False
            
            # Calculate total distance including physical tracking
            physical_distance = 0.0
            if self._exit_state and self._exit_state.trajectory_before_exit:
                traj = self._exit_state.trajectory_before_exit
                if len(traj) >= 2:
                    start = traj[0]
                    end = traj[-1]
                    physical_distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            
            virtual_distance = self._virtual_ball.distance_traveled
            total_distance = physical_distance + virtual_distance
            
            # Convert to real-world units for clearer logging
            ppm = self._current_pixels_per_meter
            physical_cm = (physical_distance / ppm) * 100
            virtual_cm = (virtual_distance / ppm) * 100
            total_cm = (total_distance / ppm) * 100
            expected_total_cm = (self._expected_total_distance_px / ppm) * 100 if self._expected_total_distance_px > 0 else 0
            expected_virtual_cm = (self._expected_virtual_distance_px / ppm) * 100 if self._expected_virtual_distance_px > 0 else 0
            
            logger.info(f"=== VIRTUAL ROLLING FINISHED ===")
            logger.info(f"  Physical distance: {physical_distance:.1f} px ({physical_cm:.1f} cm)")
            logger.info(f"  Virtual distance: {virtual_distance:.1f} px ({virtual_cm:.1f} cm) [expected: {expected_virtual_cm:.1f} cm]")
            logger.info(f"  TOTAL distance: {total_distance:.1f} px ({total_cm:.1f} cm) [expected: {expected_total_cm:.1f} cm]")
            logger.info(f"  Roll time: {self._virtual_ball.time_since_exit:.2f}s")
            logger.info(f"  Final position: ({self._virtual_ball.x:.1f}, {self._virtual_ball.y:.1f})")
        
        self._transition_to_stopped(frame_id)
