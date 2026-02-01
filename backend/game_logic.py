"""
Game logic for putting simulator.
Handles hole result calculation, scoring, and game state.
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class ShotResult(Enum):
    """Possible results of a putt."""
    PENDING = "pending"           # Shot in progress
    MADE = "made"                 # Ball went in the hole
    MISS_SHORT = "miss_short"     # Stopped before the hole
    MISS_LONG = "miss_long"       # Rolled past the hole
    MISS_LEFT = "miss_left"       # Missed to the left
    MISS_RIGHT = "miss_right"     # Missed to the right
    LIP_OUT = "lip_out"           # Hit the edge but didn't drop


# Standard golf hole diameter: 4.25 inches = 108mm = 0.108m
HOLE_DIAMETER_M = 0.108
HOLE_RADIUS_M = HOLE_DIAMETER_M / 2  # 0.054m = 54mm

# Ball must be traveling slow enough to drop into the hole
# Based on physics, ~1.5 m/s is the maximum speed to drop
MAX_SPEED_TO_DROP_M_S = 1.5

# Tolerances for determining miss direction
# If ball is within this lateral distance from hole center, it's short/long
LATERAL_TOLERANCE_M = 0.10  # 10cm


@dataclass
class HoleConfig:
    """Configuration for the hole/target."""
    distance_m: float = 3.0           # Distance from ball starting position to hole
    position_x_m: float = 0.0         # Lateral position (0 = center)
    position_y_m: float = 3.0         # Forward position (distance from start)
    radius_m: float = HOLE_RADIUS_M   # Hole radius
    
    def get_position(self) -> Tuple[float, float]:
        """Get hole position as (x, y) in meters."""
        return (self.position_x_m, self.position_y_m)


@dataclass
class ShotAnalysis:
    """Analysis of a completed shot."""
    result: ShotResult
    distance_to_hole_m: float         # Final distance from ball to hole center
    lateral_miss_m: float             # Lateral miss distance (+ = right, - = left)
    depth_miss_m: float               # Depth miss (+ = long, - = short)
    speed_at_hole_m_s: float          # Ball speed when passing the hole
    was_on_line: bool                 # Whether ball was heading toward hole
    would_have_made_if_speed: bool    # Would have made if speed was correct
    
    @property
    def is_made(self) -> bool:
        return self.result == ShotResult.MADE
    
    @property
    def miss_description(self) -> str:
        """Human-readable description of the miss."""
        if self.result == ShotResult.MADE:
            return "MADE IT!"
        elif self.result == ShotResult.MISS_SHORT:
            return f"Short by {abs(self.depth_miss_m * 100):.0f}cm"
        elif self.result == ShotResult.MISS_LONG:
            return f"Long by {abs(self.depth_miss_m * 100):.0f}cm"
        elif self.result == ShotResult.MISS_LEFT:
            return f"Left by {abs(self.lateral_miss_m * 100):.0f}cm"
        elif self.result == ShotResult.MISS_RIGHT:
            return f"Right by {abs(self.lateral_miss_m * 100):.0f}cm"
        elif self.result == ShotResult.LIP_OUT:
            return "Lip out!"
        else:
            return "Unknown"


class GameLogic:
    """
    Core game logic for putting simulator.
    Calculates shot results, manages hole configuration, and provides scoring.
    """
    
    def __init__(self, hole_distance_m: float = 3.0):
        """
        Initialize game logic.
        
        Args:
            hole_distance_m: Distance from ball starting position to hole center
        """
        self.hole = HoleConfig(
            distance_m=hole_distance_m,
            position_y_m=hole_distance_m
        )
        self._last_analysis: Optional[ShotAnalysis] = None
    
    def set_hole_distance(self, distance_m: float):
        """Update the hole distance."""
        self.hole.distance_m = distance_m
        self.hole.position_y_m = distance_m
        logger.info(f"Hole distance set to {distance_m:.2f}m")
    
    def get_hole_distance(self) -> float:
        """Get current hole distance in meters."""
        return self.hole.distance_m
    
    def analyze_shot(
        self,
        final_x_m: float,
        final_y_m: float,
        final_speed_m_s: float,
        direction_deg: float,
        initial_x_m: float = 0.0,
        initial_y_m: float = 0.0
    ) -> ShotAnalysis:
        """
        Analyze a completed shot and determine the result.
        
        Coordinate system:
        - Origin (0, 0) is the ball starting position
        - +Y is forward (toward the hole)
        - +X is to the right
        - Direction is in degrees, 0 = straight ahead, + = right, - = left
        
        Args:
            final_x_m: Final ball X position in meters
            final_y_m: Final ball Y position in meters (distance traveled)
            final_speed_m_s: Ball speed when it stopped (should be ~0)
            direction_deg: Initial direction of the shot
            initial_x_m: Starting X position (default 0)
            initial_y_m: Starting Y position (default 0)
        
        Returns:
            ShotAnalysis with the result and details
        """
        hole_x, hole_y = self.hole.get_position()
        hole_radius = self.hole.radius_m
        
        # Calculate distance from final ball position to hole center
        dx = final_x_m - hole_x
        dy = final_y_m - hole_y
        distance_to_hole = math.sqrt(dx * dx + dy * dy)
        
        # Lateral and depth miss
        lateral_miss = dx  # + = right of hole, - = left
        depth_miss = dy    # + = past hole (long), - = short of hole
        
        # Calculate speed at the hole position (estimate based on trajectory)
        # For now, use final speed since we're checking when ball stops
        speed_at_hole = final_speed_m_s
        
        # Was the ball heading toward the hole?
        # Simple check: if lateral miss is small, it was on line
        was_on_line = abs(lateral_miss) < hole_radius + 0.05  # 5cm tolerance
        
        # Determine the result
        result = ShotResult.PENDING
        
        if distance_to_hole <= hole_radius:
            # Ball ended up in/on the hole
            if final_speed_m_s <= MAX_SPEED_TO_DROP_M_S:
                result = ShotResult.MADE
            else:
                # Moving too fast - would lip out
                result = ShotResult.LIP_OUT
        elif abs(lateral_miss) <= LATERAL_TOLERANCE_M:
            # On line but missed - either short or long
            if depth_miss < 0:
                result = ShotResult.MISS_SHORT
            else:
                result = ShotResult.MISS_LONG
        else:
            # Off line - left or right
            if lateral_miss < 0:
                result = ShotResult.MISS_LEFT
            else:
                result = ShotResult.MISS_RIGHT
        
        # Would have made if speed was correct?
        # Check if ball path crossed through the hole area
        would_have_made = (
            abs(lateral_miss) <= hole_radius and 
            (depth_miss >= -hole_radius)  # Ball reached the hole area
        )
        
        analysis = ShotAnalysis(
            result=result,
            distance_to_hole_m=distance_to_hole,
            lateral_miss_m=lateral_miss,
            depth_miss_m=depth_miss,
            speed_at_hole_m_s=speed_at_hole,
            was_on_line=was_on_line,
            would_have_made_if_speed=would_have_made
        )
        
        self._last_analysis = analysis
        
        logger.info(
            f"Shot analysis: {result.value}, "
            f"distance_to_hole={distance_to_hole:.3f}m, "
            f"lateral={lateral_miss:.3f}m, depth={depth_miss:.3f}m"
        )
        
        return analysis
    
    def analyze_shot_from_backend_state(
        self,
        shot_data: dict,
        pixels_per_meter: float
    ) -> Optional[ShotAnalysis]:
        """
        Analyze a shot using data from the backend state message.
        
        Args:
            shot_data: The 'shot' object from backend state
            pixels_per_meter: Current calibration value
        
        Returns:
            ShotAnalysis or None if data is insufficient
        """
        if not shot_data:
            return None
        
        # Get total distance traveled
        distance_m = shot_data.get('distance_m', 0)
        direction_deg = shot_data.get('direction_deg', 0)
        speed_m_s = shot_data.get('speed_m_s', 0)
        
        # Convert direction to final position
        # Assuming ball starts at (0, 0)
        direction_rad = math.radians(direction_deg)
        final_x_m = distance_m * math.sin(direction_rad)  # Lateral displacement
        final_y_m = distance_m * math.cos(direction_rad)  # Forward distance
        
        return self.analyze_shot(
            final_x_m=final_x_m,
            final_y_m=final_y_m,
            final_speed_m_s=0.0,  # Ball has stopped
            direction_deg=direction_deg
        )
    
    def get_last_analysis(self) -> Optional[ShotAnalysis]:
        """Get the last shot analysis."""
        return self._last_analysis
    
    def get_state_for_websocket(self) -> dict:
        """
        Get game logic state for WebSocket broadcast.
        
        Returns:
            Dictionary with game logic state
        """
        analysis = self._last_analysis
        
        return {
            "hole": {
                "distance_m": self.hole.distance_m,
                "position_x_m": self.hole.position_x_m,
                "position_y_m": self.hole.position_y_m,
                "radius_m": self.hole.radius_m,
            },
            "last_shot": {
                "result": analysis.result.value if analysis else None,
                "distance_to_hole_m": round(analysis.distance_to_hole_m, 3) if analysis else None,
                "lateral_miss_m": round(analysis.lateral_miss_m, 3) if analysis else None,
                "depth_miss_m": round(analysis.depth_miss_m, 3) if analysis else None,
                "miss_description": analysis.miss_description if analysis else None,
                "is_made": analysis.is_made if analysis else False,
            } if analysis else None
        }


# Singleton instance for easy access
_game_logic: Optional[GameLogic] = None


def get_game_logic() -> GameLogic:
    """Get the global game logic instance."""
    global _game_logic
    if _game_logic is None:
        _game_logic = GameLogic()
    return _game_logic


def set_hole_distance(distance_m: float):
    """Set the hole distance (convenience function)."""
    get_game_logic().set_hole_distance(distance_m)
