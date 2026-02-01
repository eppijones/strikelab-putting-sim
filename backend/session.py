"""
Session tracking for putting simulator.
Tracks putts made/attempted, streaks, and session statistics.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

from .game_logic import ShotResult, ShotAnalysis

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_database = None

def _get_db():
    """Get database instance lazily."""
    global _database
    if _database is None:
        try:
            from .database import get_database
            _database = get_database()
        except Exception as e:
            logger.warning(f"Could not initialize database: {e}")
            _database = False  # Mark as failed
    return _database if _database else None


@dataclass
class ShotRecord:
    """Record of a single shot."""
    timestamp: float
    speed_m_s: float
    distance_m: float
    direction_deg: float
    target_distance_m: float
    result: ShotResult
    distance_to_hole_m: float
    lateral_miss_m: float
    depth_miss_m: float
    
    @property
    def is_made(self) -> bool:
        return self.result == ShotResult.MADE


@dataclass
class SessionStats:
    """Aggregate statistics for a session."""
    total_putts: int = 0
    putts_made: int = 0
    current_streak: int = 0
    best_streak: int = 0
    
    # By distance band (in meters)
    putts_by_distance: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Averages
    avg_speed_m_s: float = 0.0
    avg_miss_distance_m: float = 0.0
    avg_lateral_miss_m: float = 0.0
    
    @property
    def make_percentage(self) -> float:
        """Calculate make percentage."""
        if self.total_putts == 0:
            return 0.0
        return (self.putts_made / self.total_putts) * 100
    
    @property
    def miss_percentage(self) -> float:
        """Calculate miss percentage."""
        return 100.0 - self.make_percentage


class SessionManager:
    """
    Manages session state and statistics.
    Tracks all putts in the current session and computes statistics.
    """
    
    # Distance bands for categorization
    DISTANCE_BANDS = [
        ("0-1m", 0.0, 1.0),
        ("1-2m", 1.0, 2.0),
        ("2-3m", 2.0, 3.0),
        ("3-4m", 3.0, 4.0),
        ("4-5m", 4.0, 5.0),
        ("5-6m", 5.0, 6.0),
        ("6m+", 6.0, float('inf')),
    ]
    
    def __init__(self):
        """Initialize a new session."""
        self.session_id = datetime.utcnow().isoformat()
        self.start_time = time.time()
        self.shots: List[ShotRecord] = []
        self._current_streak = 0
        self._best_streak = 0
    
    def record_shot(self, analysis: ShotAnalysis, shot_data: dict, target_distance_m: float) -> ShotRecord:
        """
        Record a completed shot.
        
        Args:
            analysis: The shot analysis from GameLogic
            shot_data: Raw shot data from backend
            target_distance_m: The target hole distance
        
        Returns:
            The created ShotRecord
        """
        record = ShotRecord(
            timestamp=time.time(),
            speed_m_s=shot_data.get('speed_m_s', 0),
            distance_m=shot_data.get('distance_m', 0),
            direction_deg=shot_data.get('direction_deg', 0),
            target_distance_m=target_distance_m,
            result=analysis.result,
            distance_to_hole_m=analysis.distance_to_hole_m,
            lateral_miss_m=analysis.lateral_miss_m,
            depth_miss_m=analysis.depth_miss_m
        )
        
        self.shots.append(record)
        
        # Update streak
        if record.is_made:
            self._current_streak += 1
            if self._current_streak > self._best_streak:
                self._best_streak = self._current_streak
        else:
            self._current_streak = 0
        
        # Persist to database
        db = _get_db()
        if db:
            try:
                db.save_shot(
                    session_id=self.session_id,
                    speed_m_s=record.speed_m_s,
                    distance_m=record.distance_m,
                    direction_deg=record.direction_deg,
                    target_distance_m=record.target_distance_m,
                    result=record.result,
                    distance_to_hole_m=record.distance_to_hole_m,
                    lateral_miss_m=record.lateral_miss_m,
                    depth_miss_m=record.depth_miss_m
                )
            except Exception as e:
                logger.error(f"Failed to persist shot to database: {e}")
        
        logger.info(
            f"Shot recorded: {analysis.result.value}, "
            f"total={len(self.shots)}, made={self.get_putts_made()}, "
            f"streak={self._current_streak}"
        )
        
        return record
    
    def get_putts_made(self) -> int:
        """Get number of putts made."""
        return sum(1 for shot in self.shots if shot.is_made)
    
    def get_total_putts(self) -> int:
        """Get total number of putts."""
        return len(self.shots)
    
    def get_make_percentage(self) -> float:
        """Get make percentage."""
        total = self.get_total_putts()
        if total == 0:
            return 0.0
        return (self.get_putts_made() / total) * 100
    
    def get_current_streak(self) -> int:
        """Get current consecutive makes."""
        return self._current_streak
    
    def get_best_streak(self) -> int:
        """Get best consecutive makes in session."""
        return self._best_streak
    
    def get_stats(self) -> SessionStats:
        """Calculate and return session statistics."""
        stats = SessionStats(
            total_putts=self.get_total_putts(),
            putts_made=self.get_putts_made(),
            current_streak=self._current_streak,
            best_streak=self._best_streak,
        )
        
        if not self.shots:
            return stats
        
        # Calculate averages
        speeds = [s.speed_m_s for s in self.shots]
        stats.avg_speed_m_s = sum(speeds) / len(speeds) if speeds else 0.0
        
        misses = [s for s in self.shots if not s.is_made]
        if misses:
            stats.avg_miss_distance_m = sum(s.distance_to_hole_m for s in misses) / len(misses)
            stats.avg_lateral_miss_m = sum(abs(s.lateral_miss_m) for s in misses) / len(misses)
        
        # Calculate by distance band
        for band_name, min_dist, max_dist in self.DISTANCE_BANDS:
            band_shots = [s for s in self.shots if min_dist <= s.target_distance_m < max_dist]
            if band_shots:
                made = sum(1 for s in band_shots if s.is_made)
                stats.putts_by_distance[band_name] = {
                    "total": len(band_shots),
                    "made": made,
                    "percentage": round((made / len(band_shots)) * 100, 1)
                }
        
        return stats
    
    def get_last_n_shots(self, n: int = 10) -> List[ShotRecord]:
        """Get the last N shots."""
        return self.shots[-n:] if self.shots else []
    
    def reset(self):
        """Reset the session."""
        # End previous session in database
        db = _get_db()
        if db and self.shots:
            try:
                db.end_session(
                    session_id=self.session_id,
                    total_putts=self.get_total_putts(),
                    putts_made=self.get_putts_made(),
                    best_streak=self._best_streak
                )
            except Exception as e:
                logger.error(f"Failed to end session in database: {e}")
        
        # Start new session
        self.session_id = datetime.utcnow().isoformat()
        self.start_time = time.time()
        self.shots = []
        self._current_streak = 0
        self._best_streak = 0
        
        # Record new session start
        if db:
            try:
                db.start_session(self.session_id)
            except Exception as e:
                logger.error(f"Failed to start session in database: {e}")
        
        logger.info("Session reset")
    
    def get_state_for_websocket(self) -> dict:
        """
        Get session state for WebSocket broadcast.
        
        Returns:
            Dictionary with session state
        """
        stats = self.get_stats()
        
        return {
            "session_id": self.session_id,
            "duration_s": round(time.time() - self.start_time, 1),
            "total_putts": stats.total_putts,
            "putts_made": stats.putts_made,
            "make_percentage": round(stats.make_percentage, 1),
            "current_streak": stats.current_streak,
            "best_streak": stats.best_streak,
            "avg_speed_m_s": round(stats.avg_speed_m_s, 2),
            "avg_miss_distance_m": round(stats.avg_miss_distance_m, 3),
            "putts_by_distance": stats.putts_by_distance,
        }


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def reset_session():
    """Reset the current session."""
    get_session_manager().reset()
