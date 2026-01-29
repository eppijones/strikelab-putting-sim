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
    virtual_distance_px: float = 0.0   # Distance traveled virtually (after frame exit)
    total_distance_px: float = 0.0     # Total distance (physical + virtual)
    exited_frame: bool = False         # Whether ball exited the camera view
    

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
    VELOCITY_WINDOW = 3             # Frames for initial velocity computation (fast)
    VELOCITY_WINDOW_STABLE = 5      # Frames for stable velocity computation
    ROI_PADDING = 100               # Pixels to pad around ball for ROI tracking
    
    # Background model parameters
    BG_DELTA_THRESHOLD = 30.0       # Foreground delta to trigger motion
    BG_LEARNING_RATE = 0.01         # Background update rate
    
    # False shot prevention
    MAX_RADIUS_CHANGE_RATIO = 0.5   # Reject if radius changes > 50%
    MIN_CONFIDENCE_THRESHOLD = 0.7  # Minimum detection confidence
    MAX_POSITION_JUMP_PX = 100      # Large jump = ball placement, not shot
    MIN_SHOT_FRAMES = 5             # Minimum frames for valid shot
    
    # Frame exit detection - for virtual ball continuation
    FRAME_EXIT_MARGIN_PX = 30       # Ball considered "exited" when this close to edge
    MIN_EXIT_SPEED_PX_S = 100       # Minimum speed to trigger virtual rolling
    MIN_TRACKING_FRAMES_FOR_EXIT = 10  # Need enough data for curve estimation
    
    # Virtual ball physics - LINEAR DECELERATION MODEL
    # Real putting physics: constant friction force → constant deceleration
    # v(t) = v0 - a*t, stops when v=0, distance = v0²/(2a)
    # 
    # Default deceleration: 0.55 m/s² (configurable via config.json)
    # Stored in m/s² and converted to px/s² using current calibration
    DECELERATION_M_S2 = 0.55        # Default deceleration in m/s²
    MIN_VIRTUAL_SPEED_PX_S = 20     # Stop virtual rolling below this speed
    MAX_VIRTUAL_TIME_S = 10.0       # Maximum virtual rolling time (safety limit)
    
    # Current calibration for pixel/meter conversion
    _current_pixels_per_meter: float = 1150.0  # Updated by set_calibration()
    
    def __init__(
        self, 
        detector: Optional[BallDetector] = None,
        motion_threshold_px: Optional[float] = None,
        motion_confirm_frames: Optional[int] = None,
        stopped_velocity_threshold: Optional[float] = None,
        stopped_confirm_frames: Optional[int] = None,
        cooldown_duration_ms: Optional[int] = None,
        idle_ema_alpha: Optional[float] = None,
        deceleration_px_s2: Optional[float] = None
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
        
        # Position tracking
        self._current_pos: Optional[Tuple[float, float]] = None
        self._smoothed_pos: Optional[Tuple[float, float]] = None
        self._locked_pos: Optional[Tuple[float, float]] = None  # Stillness lock position
        self._last_radius: Optional[float] = None
        self._last_confidence: float = 0.0
        
        # Trajectory recording
        self._trajectory: deque[TrackPoint] = deque(maxlen=1000)
        self._raw_trajectory: deque[TrackPoint] = deque(maxlen=20)  # Raw positions for velocity
        self._motion_start_frame: int = 0
        self._first_speed_frame: int = 0
        
        # Motion detection
        self._motion_trigger_count = 0
        self._stopped_count = 0
        self._cooldown_start_ns: int = 0
        self._settling_countdown: int = 0  # Frames to wait before motion detection
        
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
            
            # Expected movement based on velocity and frame time (assume ~120fps)
            expected_move = self._velocity.speed / 120
            
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
        Compute velocity from raw trajectory points.
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
        
        # Simple two-point velocity for maximum responsiveness
        p1 = points[0]
        p2 = points[-1]
        
        dt_s = (p2.timestamp_ns - p1.timestamp_ns) / 1e9
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
        
        dt_s = (p2.timestamp_ns - p1.timestamp_ns) / 1e9
        if dt_s <= 0:
            return
        
        vx = (p2.x - p1.x) / dt_s
        vy = (p2.y - p1.y) / dt_s
        
        self._velocity = Velocity(vx=vx, vy=vy)
        self._velocity_history.append(self._velocity)
    
    def _check_transitions(self, detection: Detection, timestamp_ns: int, frame_id: int):
        """Check for state machine transitions."""
        if self._state == ShotState.ARMED:
            self._check_armed_to_tracking(detection, frame_id)
        elif self._state == ShotState.TRACKING:
            self._check_tracking_to_stopped(frame_id)
        elif self._state == ShotState.STOPPED:
            self._transition_to_cooldown(timestamp_ns)
    
    def _check_armed_to_tracking(self, detection: Detection, frame_id: int):
        """
        Check if motion threshold exceeded to start tracking.
        
        Uses displacement from locked position.
        Requires MOTION_CONFIRM_FRAMES consecutive triggers to prevent false shots.
        Must maintain motion direction (not just jitter in place).
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
        displacement = np.sqrt(
            (new_pos[0] - ref_pos[0])**2 +
            (new_pos[1] - ref_pos[1])**2
        )
        
        # Motion triggered only by significant displacement
        motion_detected = displacement > self.MOTION_THRESHOLD_PX
        
        if motion_detected:
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
                f"displacement={displacement:.1f}px, accumulated={accumulated_dist:.1f}px"
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
        # This is where the ball was when the putt started
        self._shot_start_pos = self._current_pos
        
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
        
        logger.info(f"Shot started at position: {self._shot_start_pos}")
    
    def _check_tracking_to_stopped(self, frame_id: int):
        """Check if ball has stopped moving."""
        # Don't transition to STOPPED if we recently lost detection but had high velocity
        # This indicates ball might have exited frame, not stopped
        if self._frames_lost > 0 and self._last_valid_velocity:
            if self._last_valid_velocity.speed > self.MIN_EXIT_SPEED_PX_S:
                # Ball was moving fast when we lost it - don't call it "stopped"
                # Frame exit detection should handle this
                logger.debug(f"Ignoring stop check: frames_lost={self._frames_lost}, "
                           f"last_speed={self._last_valid_velocity.speed:.1f}")
                return
        
        if self._velocity is None or self._velocity.speed < self.STOPPED_VELOCITY_THRESHOLD:
            self._stopped_count += 1
            if self._stopped_count >= self.STOPPED_CONFIRM_FRAMES:
                self._transition_to_stopped(frame_id)
        else:
            self._stopped_count = 0
    
    def _transition_to_stopped(self, frame_id: int):
        """Transition from TRACKING to STOPPED."""
        logger.info(f"TRACKING -> STOPPED at frame {frame_id}")
        self._state = ShotState.STOPPED
        
        # Compute shot result
        self._compute_shot_result(frame_id)
    
    def _compute_shot_result(self, frame_id: int):
        """Compute final shot metrics."""
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
        
        # Physical distance (trajectory in frame)
        physical_distance_px = 0.0
        if trajectory_points:
            start_pos = trajectory_points[0]
            end_pos = trajectory_points[-1]
            physical_distance_px = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Virtual distance (after frame exit)
        virtual_distance_px = 0.0
        exited_frame = False
        if self._virtual_ball:
            virtual_distance_px = self._virtual_ball.distance_traveled
            exited_frame = True
        
        # Total distance
        total_distance_px = physical_distance_px + virtual_distance_px
        
        # Log trajectory info for debugging
        logger.info(f"Trajectory: {len(trajectory_points)} points, "
                   f"physical={physical_distance_px:.1f}px, virtual={virtual_distance_px:.1f}px, "
                   f"total={total_distance_px:.1f}px")
        
        self._shot_result = ShotResult(
            initial_speed_px_s=initial_speed,
            initial_direction_deg=initial_direction,
            frames_to_tracking=frames_to_tracking,
            frames_to_speed=frames_to_speed,
            trajectory=trajectory_points,  # Full trajectory (no longer slicing)
            duration_ms=duration_ms,
            virtual_distance_px=virtual_distance_px,
            total_distance_px=total_distance_px,
            exited_frame=exited_frame
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
    
    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        """Get current trajectory as list of (x, y) tuples."""
        return [(p.x, p.y) for p in self._trajectory]
    
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
        """Transition from TRACKING to VIRTUAL_ROLLING when ball exits frame."""
        logger.info(f"TRACKING -> VIRTUAL_ROLLING at frame {frame_id}")
        
        # Capture exit state
        exit_pos = self._last_valid_position or self._current_pos
        exit_vel = self._last_valid_velocity or self._velocity
        
        if exit_pos is None or exit_vel is None:
            logger.error("Cannot transition to virtual rolling: missing position/velocity")
            self._transition_to_stopped(frame_id)
            return
        
        curvature = self._compute_trajectory_curvature()
        
        self._exit_state = ExitState(
            position=exit_pos,
            velocity=(exit_vel.vx, exit_vel.vy),
            speed=exit_vel.speed,
            direction_deg=exit_vel.direction_deg,
            curvature=curvature,
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            trajectory_before_exit=[(p.x, p.y) for p in self._trajectory]
        )
        
        # Calculate expected final distance using LINEAR physics: distance = v0²/(2a)
        a = self.get_deceleration_px_s2()
        expected_max_distance = (exit_vel.speed ** 2) / (2 * a)
        expected_stop_time = exit_vel.speed / a
        
        # Initialize virtual ball at exit position
        self._virtual_ball = VirtualBallState(
            x=exit_pos[0],
            y=exit_pos[1],
            vx=exit_vel.vx,
            vy=exit_vel.vy,
            speed=exit_vel.speed,
            distance_traveled=0.0,
            time_since_exit=0.0,
            is_rolling=True,
            final_position=None
        )
        
        self._virtual_start_time_ns = timestamp_ns
        self._state = ShotState.VIRTUAL_ROLLING
        
        logger.info(f"=== VIRTUAL ROLLING STARTED ===")
        logger.info(f"  Exit position: {exit_pos}")
        logger.info(f"  Exit speed: {exit_vel.speed:.1f} px/s")
        logger.info(f"  Exit direction: {exit_vel.direction_deg:.1f}°")
        logger.info(f"  Curvature: {curvature:.4f}°/px")
        logger.info(f"  Deceleration: {a:.1f} px/s²")
        logger.info(f"  Expected distance: {expected_max_distance:.1f} px")
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
        a = self.get_deceleration_px_s2()
        
        # Time when ball stops
        t_stop = initial_speed / a
        
        # Check if ball has stopped
        if dt >= t_stop:
            # Ball has stopped - use final values
            current_speed = 0.0
            distance = (initial_speed ** 2) / (2 * a)  # Final distance
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
        max_distance = (initial_speed ** 2) / (2 * a)
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
            
            logger.info(f"=== VIRTUAL ROLLING FINISHED ===")
            logger.info(f"  Physical distance: {physical_distance:.1f} px")
            logger.info(f"  Virtual distance: {virtual_distance:.1f} px")
            logger.info(f"  TOTAL distance: {total_distance:.1f} px")
            logger.info(f"  Roll time: {self._virtual_ball.time_since_exit:.2f}s")
            logger.info(f"  Final position: ({self._virtual_ball.x:.1f}, {self._virtual_ball.y:.1f})")
        
        self._transition_to_stopped(frame_id)