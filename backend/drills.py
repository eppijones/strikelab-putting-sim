"""
Practice drill modes for putting simulator.
Includes distance control, gate drills, ladder drills, etc.
"""

import logging
import random
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DrillType(Enum):
    """Available drill types."""
    NONE = "none"
    DISTANCE_CONTROL = "distance_control"
    GATE_DRILL = "gate_drill"
    LADDER_DRILL = "ladder_drill"


class ScoreRating(Enum):
    """Score rating for a drill attempt."""
    PERFECT = "perfect"     # < 10cm from target
    GREAT = "great"         # < 20cm
    GOOD = "good"           # < 40cm
    FAIR = "fair"           # < 60cm
    MISS = "miss"           # > 60cm


@dataclass
class DrillAttempt:
    """Record of a single drill attempt."""
    timestamp: float
    target_distance_m: float
    actual_distance_m: float
    error_m: float
    rating: ScoreRating
    points: int
    direction_deg: float


@dataclass
class DrillSession:
    """State for an active drill session."""
    drill_type: DrillType
    start_time: float
    attempts: List[DrillAttempt] = field(default_factory=list)
    total_points: int = 0
    current_target_m: float = 0.0
    targets_completed: int = 0
    
    # Drill-specific state
    ladder_position: int = 0  # For ladder drill


class DistanceControlDrill:
    """
    Distance control drill - practice hitting specific distances.
    
    Rules:
    - Random target distances are generated (2-6m range)
    - Score based on how close you get to the target
    - Perfect: < 10cm = 100 points
    - Great: < 20cm = 75 points
    - Good: < 40cm = 50 points
    - Fair: < 60cm = 25 points
    - Miss: > 60cm = 0 points
    """
    
    # Scoring thresholds (in meters)
    PERFECT_THRESHOLD = 0.10
    GREAT_THRESHOLD = 0.20
    GOOD_THRESHOLD = 0.40
    FAIR_THRESHOLD = 0.60
    
    # Point values
    POINTS = {
        ScoreRating.PERFECT: 100,
        ScoreRating.GREAT: 75,
        ScoreRating.GOOD: 50,
        ScoreRating.FAIR: 25,
        ScoreRating.MISS: 0,
    }
    
    def __init__(self, min_distance: float = 2.0, max_distance: float = 6.0):
        """
        Initialize distance control drill.
        
        Args:
            min_distance: Minimum target distance in meters
            max_distance: Maximum target distance in meters
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
    
    def generate_target(self) -> float:
        """Generate a random target distance."""
        # Round to nearest 0.5m for cleaner targets
        target = random.uniform(self.min_distance, self.max_distance)
        return round(target * 2) / 2
    
    def score_attempt(self, target_m: float, actual_m: float) -> tuple[ScoreRating, int]:
        """
        Score an attempt based on distance from target.
        
        Args:
            target_m: Target distance in meters
            actual_m: Actual distance achieved in meters
        
        Returns:
            Tuple of (rating, points)
        """
        error = abs(actual_m - target_m)
        
        if error <= self.PERFECT_THRESHOLD:
            rating = ScoreRating.PERFECT
        elif error <= self.GREAT_THRESHOLD:
            rating = ScoreRating.GREAT
        elif error <= self.GOOD_THRESHOLD:
            rating = ScoreRating.GOOD
        elif error <= self.FAIR_THRESHOLD:
            rating = ScoreRating.FAIR
        else:
            rating = ScoreRating.MISS
        
        return rating, self.POINTS[rating]


class LadderDrill:
    """
    Ladder drill - progressive distance control.
    
    Rules:
    - Start at 1m, progress through: 1m, 2m, 3m, 4m, 5m, 6m
    - Must make each distance (within tolerance) to advance
    - Miss = restart from beginning
    - Tolerance: 30cm from target
    """
    
    DISTANCES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    TOLERANCE = 0.30  # 30cm
    
    def get_current_target(self, position: int) -> float:
        """Get the target for the current ladder position."""
        if position < len(self.DISTANCES):
            return self.DISTANCES[position]
        return self.DISTANCES[-1]
    
    def check_advance(self, target_m: float, actual_m: float) -> tuple[bool, bool]:
        """
        Check if the attempt advances to next rung.
        
        Returns:
            Tuple of (advanced, completed_ladder)
        """
        error = abs(actual_m - target_m)
        advanced = error <= self.TOLERANCE
        
        # Find current position
        try:
            position = self.DISTANCES.index(target_m)
            completed = advanced and position == len(self.DISTANCES) - 1
        except ValueError:
            completed = False
        
        return advanced, completed


class DrillManager:
    """
    Manages drill sessions and scoring.
    """
    
    def __init__(self):
        self.session: Optional[DrillSession] = None
        self.distance_drill = DistanceControlDrill()
        self.ladder_drill = LadderDrill()
    
    def start_drill(self, drill_type: DrillType) -> dict:
        """
        Start a new drill session.
        
        Returns:
            Initial drill state
        """
        self.session = DrillSession(
            drill_type=drill_type,
            start_time=time.time()
        )
        
        if drill_type == DrillType.DISTANCE_CONTROL:
            self.session.current_target_m = self.distance_drill.generate_target()
        elif drill_type == DrillType.LADDER_DRILL:
            self.session.ladder_position = 0
            self.session.current_target_m = self.ladder_drill.get_current_target(0)
        
        logger.info(f"Started {drill_type.value} drill, target={self.session.current_target_m}m")
        
        return self.get_state()
    
    def stop_drill(self) -> dict:
        """Stop the current drill and return final stats."""
        if not self.session:
            return {"drill_type": DrillType.NONE.value}
        
        final_state = self.get_state()
        self.session = None
        
        return final_state
    
    def record_attempt(self, actual_distance_m: float, direction_deg: float) -> dict:
        """
        Record a drill attempt and update state.
        
        Args:
            actual_distance_m: Distance achieved
            direction_deg: Direction of the putt
        
        Returns:
            Result of the attempt
        """
        if not self.session:
            return {"error": "No active drill"}
        
        target_m = self.session.current_target_m
        error_m = actual_distance_m - target_m
        
        # Score based on drill type
        if self.session.drill_type == DrillType.DISTANCE_CONTROL:
            rating, points = self.distance_drill.score_attempt(target_m, actual_distance_m)
            
            # Generate next target
            next_target = self.distance_drill.generate_target()
            self.session.current_target_m = next_target
            self.session.targets_completed += 1
            advanced = True
            reset = False
            
        elif self.session.drill_type == DrillType.LADDER_DRILL:
            advanced, completed = self.ladder_drill.check_advance(target_m, actual_distance_m)
            
            if advanced:
                points = 100
                rating = ScoreRating.PERFECT
                self.session.ladder_position += 1
                
                if completed:
                    # Completed the ladder!
                    points = 500  # Bonus for completing
                    self.session.targets_completed += 1
                    # Reset to start for another round
                    self.session.ladder_position = 0
                
                self.session.current_target_m = self.ladder_drill.get_current_target(
                    self.session.ladder_position
                )
                reset = False
            else:
                # Miss - restart ladder
                points = 0
                rating = ScoreRating.MISS
                self.session.ladder_position = 0
                self.session.current_target_m = self.ladder_drill.get_current_target(0)
                reset = True
        else:
            rating = ScoreRating.MISS
            points = 0
            advanced = False
            reset = False
        
        # Record attempt
        attempt = DrillAttempt(
            timestamp=time.time(),
            target_distance_m=target_m,
            actual_distance_m=actual_distance_m,
            error_m=error_m,
            rating=rating,
            points=points,
            direction_deg=direction_deg
        )
        self.session.attempts.append(attempt)
        self.session.total_points += points
        
        logger.info(
            f"Drill attempt: target={target_m:.1f}m, actual={actual_distance_m:.2f}m, "
            f"error={error_m:.2f}m, rating={rating.value}, points={points}"
        )
        
        return {
            "target_m": target_m,
            "actual_m": round(actual_distance_m, 2),
            "error_m": round(error_m, 2),
            "error_cm": round(abs(error_m) * 100, 1),
            "rating": rating.value,
            "points": points,
            "total_points": self.session.total_points,
            "attempts": len(self.session.attempts),
            "next_target_m": self.session.current_target_m,
            "advanced": advanced,
            "reset": reset,
            "ladder_position": self.session.ladder_position if self.session.drill_type == DrillType.LADDER_DRILL else None
        }
    
    def get_state(self) -> dict:
        """Get current drill state for WebSocket broadcast."""
        if not self.session:
            return {
                "active": False,
                "drill_type": DrillType.NONE.value
            }
        
        return {
            "active": True,
            "drill_type": self.session.drill_type.value,
            "current_target_m": self.session.current_target_m,
            "total_points": self.session.total_points,
            "attempts": len(self.session.attempts),
            "targets_completed": self.session.targets_completed,
            "duration_s": round(time.time() - self.session.start_time, 1),
            "ladder_position": self.session.ladder_position if self.session.drill_type == DrillType.LADDER_DRILL else None,
            "last_attempt": {
                "rating": self.session.attempts[-1].rating.value,
                "points": self.session.attempts[-1].points,
                "error_cm": round(abs(self.session.attempts[-1].error_m) * 100, 1)
            } if self.session.attempts else None
        }


# Singleton instance
_drill_manager: Optional[DrillManager] = None


def get_drill_manager() -> DrillManager:
    """Get the global drill manager instance."""
    global _drill_manager
    if _drill_manager is None:
        _drill_manager = DrillManager()
    return _drill_manager
