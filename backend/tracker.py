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
    ARMED = "ARMED"          # Waiting for ball, ready to detect motion
    TRACKING = "TRACKING"    # Ball in motion, recording trajectory
    STOPPED = "STOPPED"      # Ball stopped, computing final metrics
    COOLDOWN = "COOLDOWN"    # Brief pause before re-arming


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
class ShotResult:
    """Final shot metrics after ball stops."""
    initial_speed_px_s: float
    initial_direction_deg: float
    frames_to_tracking: int  # Frames from motion start to TRACKING state
    frames_to_speed: int     # Frames from impact to first stable speed
    trajectory: List[Tuple[float, float]]
    duration_ms: float
    

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
    
    def __init__(
        self, 
        detector: Optional[BallDetector] = None,
        motion_threshold_px: Optional[float] = None,
        motion_confirm_frames: Optional[int] = None,
        stopped_velocity_threshold: Optional[float] = None,
        stopped_confirm_frames: Optional[int] = None,
        cooldown_duration_ms: Optional[int] = None,
        idle_ema_alpha: Optional[float] = None
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
        
        # Handle state-specific logic
        if self._state == ShotState.COOLDOWN:
            self._handle_cooldown(timestamp_ns)
        
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
        
        # Duration
        if len(self._trajectory) >= 2:
            start_t = self._trajectory[0].timestamp_ns
            end_t = self._trajectory[-1].timestamp_ns
            duration_ms = (end_t - start_t) / 1e6
        else:
            duration_ms = 0.0
        
        # Log trajectory info for debugging
        if trajectory_points:
            start_pos = trajectory_points[0]
            end_pos = trajectory_points[-1]
            distance_px = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            logger.info(f"Trajectory: {len(trajectory_points)} points, "
                       f"start={start_pos}, end={end_pos}, distance={distance_px:.1f}px")
        
        self._shot_result = ShotResult(
            initial_speed_px_s=initial_speed,
            initial_direction_deg=initial_direction,
            frames_to_tracking=frames_to_tracking,
            frames_to_speed=frames_to_speed,
            trajectory=trajectory_points,  # Full trajectory (no longer slicing)
            duration_ms=duration_ms
        )
        
        logger.info(f"Shot result: speed={initial_speed:.1f}px/s, dir={initial_direction:.1f}°, "
                   f"frames_to_speed={frames_to_speed}, trajectory_points={len(trajectory_points)}")
    
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
    
    def _compute_idle_stddev(self) -> float:
        """Compute position standard deviation when idle."""
        if len(self._idle_positions) < 10:
            return 0.0
        
        positions = np.array(list(self._idle_positions))
        return float(np.std(positions, axis=0).mean())
    
    def _build_state(self) -> TrackerState:
        """Build current state for external consumption."""
        return TrackerState(
            state=self._state,
            lane=self._lane,
            ball_x=self._current_pos[0] if self._current_pos else None,
            ball_y=self._current_pos[1] if self._current_pos else None,
            ball_radius=self._last_radius,
            ball_confidence=self._last_confidence,
            velocity=self._velocity,
            shot_result=self._shot_result,
            idle_stddev=self._compute_idle_stddev()
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
        logger.info("Tracker reset")
    
    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        """Get current trajectory as list of (x, y) tuples."""
        return [(p.x, p.y) for p in self._trajectory]
