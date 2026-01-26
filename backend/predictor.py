"""
Ball trajectory predictor for StrikeLab Putting Sim.
Predicts ball path when it exits the camera view.
"""

import numpy as np
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PredictedPoint:
    """Single point in predicted trajectory."""
    x: float
    y: float
    t: float  # Time from prediction start (seconds)
    speed: float  # Speed at this point


@dataclass
class PredictionResult:
    """Result of trajectory prediction."""
    trajectory: List[PredictedPoint]
    final_position: Tuple[float, float]
    final_time: float  # Time until ball stops
    initial_speed: float
    exit_position: Tuple[float, float]


class BallPredictor:
    """
    Predicts ball trajectory after it exits the camera view.
    
    Uses exponential deceleration model:
        v(t) = v0 * e^(-k*t)
        
    where k is the friction coefficient.
    
    Integrating gives position:
        x(t) = x0 + (v0/k) * (1 - e^(-k*t))
    """
    
    def __init__(
        self,
        friction_coefficient: float = 0.15,
        min_velocity_threshold: float = 10.0,  # px/s
        max_prediction_time_s: float = 10.0,
        time_step_s: float = 0.05  # 50ms steps for trajectory
    ):
        self.friction = friction_coefficient
        self.min_velocity = min_velocity_threshold
        self.max_time = max_prediction_time_s
        self.time_step = time_step_s
    
    def predict(
        self,
        exit_position: Tuple[float, float],
        exit_velocity: Tuple[float, float],
        frame_bounds: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    ) -> Optional[PredictionResult]:
        """
        Predict ball trajectory from exit point.
        
        Args:
            exit_position: (x, y) position where ball exited view
            exit_velocity: (vx, vy) velocity at exit (pixels/second)
            frame_bounds: Optional frame bounds to check re-entry
            
        Returns:
            PredictionResult with predicted trajectory
        """
        x0, y0 = exit_position
        vx0, vy0 = exit_velocity
        
        initial_speed = np.sqrt(vx0**2 + vy0**2)
        
        if initial_speed < self.min_velocity:
            logger.debug("Exit velocity too low for prediction")
            return None
        
        # Normalize velocity direction
        if initial_speed > 0:
            dir_x = vx0 / initial_speed
            dir_y = vy0 / initial_speed
        else:
            return None
        
        # Generate trajectory points
        trajectory: List[PredictedPoint] = []
        
        t = 0.0
        while t < self.max_time:
            # Velocity at time t
            v_t = initial_speed * np.exp(-self.friction * t)
            
            if v_t < self.min_velocity:
                break
            
            # Position at time t (integrated from exponential decay)
            # Distance traveled: d(t) = (v0/k) * (1 - e^(-k*t))
            distance = (initial_speed / self.friction) * (1 - np.exp(-self.friction * t))
            
            x_t = x0 + distance * dir_x
            y_t = y0 + distance * dir_y
            
            trajectory.append(PredictedPoint(
                x=x_t,
                y=y_t,
                t=t,
                speed=v_t
            ))
            
            t += self.time_step
        
        if not trajectory:
            return None
        
        # Final position (ball stopped)
        # As t -> infinity, distance -> v0/k
        max_distance = initial_speed / self.friction
        final_x = x0 + max_distance * dir_x
        final_y = y0 + max_distance * dir_y
        
        # Time to reach min velocity
        # v(t) = v0 * e^(-k*t) = v_min
        # t = -ln(v_min/v0) / k
        final_time = -np.log(self.min_velocity / initial_speed) / self.friction
        
        return PredictionResult(
            trajectory=trajectory,
            final_position=(final_x, final_y),
            final_time=final_time,
            initial_speed=initial_speed,
            exit_position=exit_position
        )
    
    def predict_to_target(
        self,
        exit_position: Tuple[float, float],
        exit_velocity: Tuple[float, float],
        target_position: Tuple[float, float]
    ) -> Optional[dict]:
        """
        Predict if ball will reach target (hole).
        
        Returns dict with:
        - will_reach: bool
        - distance_to_target: float
        - miss_direction: float (positive = right of target)
        - time_to_closest: float
        """
        result = self.predict(exit_position, exit_velocity)
        if result is None:
            return None
        
        target_x, target_y = target_position
        final_x, final_y = result.final_position
        
        # Find closest approach to target
        min_dist = float('inf')
        closest_point = None
        time_to_closest = 0.0
        
        for point in result.trajectory:
            dist = np.sqrt((point.x - target_x)**2 + (point.y - target_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_point = (point.x, point.y)
                time_to_closest = point.t
        
        # Check final position distance
        final_dist = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
        
        if final_dist < min_dist:
            min_dist = final_dist
            closest_point = (final_x, final_y)
            time_to_closest = result.final_time
        
        # Determine miss direction (perpendicular to target direction)
        # Positive = ball passes to the right of target
        if closest_point:
            # Vector from exit to target
            to_target = np.array([target_x - exit_position[0], target_y - exit_position[1]])
            # Vector from exit to closest point
            to_closest = np.array([closest_point[0] - exit_position[0], closest_point[1] - exit_position[1]])
            
            # Cross product gives signed perpendicular distance
            cross = to_target[0] * to_closest[1] - to_target[1] * to_closest[0]
            target_dist = np.linalg.norm(to_target)
            miss_direction = cross / target_dist if target_dist > 0 else 0.0
        else:
            miss_direction = 0.0
        
        return {
            "will_reach": min_dist < 50,  # Within ~50px of target
            "distance_to_target": min_dist,
            "miss_direction": miss_direction,
            "time_to_closest": time_to_closest,
            "final_position": result.final_position,
            "trajectory": [(p.x, p.y) for p in result.trajectory]
        }
    
    def set_friction(self, friction: float):
        """Update friction coefficient."""
        self.friction = max(0.01, friction)  # Prevent division by zero
        logger.info(f"Friction coefficient set to {self.friction}")


class AdaptivePredictor(BallPredictor):
    """
    Predictor that adapts friction based on observed deceleration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._observed_frictions: List[float] = []
        self._max_observations = 20
    
    def learn_from_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]]  # (x, y, timestamp)
    ) -> Optional[float]:
        """
        Learn friction from observed trajectory.
        
        Returns estimated friction coefficient.
        """
        if len(trajectory) < 10:
            return None
        
        # Compute velocities
        velocities = []
        for i in range(1, len(trajectory)):
            x1, y1, t1 = trajectory[i-1]
            x2, y2, t2 = trajectory[i]
            
            dt = t2 - t1
            if dt <= 0:
                continue
            
            v = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / dt
            velocities.append((t1, v))
        
        if len(velocities) < 5:
            return None
        
        # Fit exponential decay
        # log(v) = log(v0) - k*t
        try:
            t_arr = np.array([v[0] for v in velocities])
            v_arr = np.array([v[1] for v in velocities])
            
            # Filter out near-zero velocities
            mask = v_arr > self.min_velocity
            if np.sum(mask) < 5:
                return None
            
            t_arr = t_arr[mask] - t_arr[mask][0]  # Normalize time
            v_arr = v_arr[mask]
            
            # Linear regression on log(v)
            log_v = np.log(v_arr)
            coeffs = np.polyfit(t_arr, log_v, 1)
            
            estimated_friction = -coeffs[0]
            
            if 0.01 < estimated_friction < 1.0:  # Sanity check
                self._observed_frictions.append(estimated_friction)
                if len(self._observed_frictions) > self._max_observations:
                    self._observed_frictions.pop(0)
                
                # Update friction to moving average
                self.friction = np.mean(self._observed_frictions)
                logger.info(f"Learned friction: {estimated_friction:.3f}, "
                           f"average: {self.friction:.3f}")
                
                return estimated_friction
                
        except Exception as e:
            logger.debug(f"Failed to learn friction: {e}")
        
        return None
