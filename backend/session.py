"""
Session tracking for putting simulator.
Tracks putts made/attempted, streaks, and session statistics.
Includes consistency metrics, tendency analysis, and miss distribution.
"""

import logging
import time
import numpy as np
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
    
    @property
    def distance_error_m(self) -> float:
        """Distance error (positive = long, negative = short)."""
        return self.distance_m - self.target_distance_m


@dataclass
class ConsistencyMetrics:
    """Consistency/variability metrics for putting analysis."""
    # Standard deviations
    speed_stddev: float = 0.0           # σ of ball speed (m/s)
    direction_stddev: float = 0.0       # σ of start line direction (degrees)
    distance_error_stddev: float = 0.0  # σ of distance error (m)
    
    # Coefficient of variation (%)
    speed_cv: float = 0.0               # CV = stddev/mean * 100
    
    # Composite consistency score (0-100, higher = more consistent)
    consistency_score: float = 0.0
    
    # Rolling consistency (last N shots)
    rolling_speed_stddev: float = 0.0
    rolling_direction_stddev: float = 0.0
    rolling_window: int = 10


@dataclass
class TendencyAnalysis:
    """Tendency/bias analysis for putting patterns."""
    # Speed/distance bias (positive = hitting long, negative = short)
    speed_bias_m_s: float = 0.0         # Avg difference from optimal speed
    distance_bias_m: float = 0.0        # Avg (actual - target) distance
    
    # Direction bias (positive = pushing right, negative = pulling left)
    direction_bias_deg: float = 0.0     # Avg direction (0 = straight)
    lateral_bias_m: float = 0.0         # Avg lateral miss (+ = right)
    
    # Dominant miss pattern
    dominant_miss: str = "none"         # "right-short", "left-long", etc.
    dominant_miss_percentage: float = 0.0
    
    # Pattern detection messages
    speed_tendency: str = ""            # "hitting long", "hitting short", "neutral"
    direction_tendency: str = ""        # "pushing right", "pulling left", "neutral"


@dataclass
class MissDistribution:
    """Distribution of misses by quadrant and type."""
    # Quadrant counts (for missed putts only)
    right_short: int = 0
    right_long: int = 0
    left_short: int = 0
    left_long: int = 0
    
    # Percentages
    right_short_pct: float = 0.0
    right_long_pct: float = 0.0
    left_short_pct: float = 0.0
    left_long_pct: float = 0.0
    
    # Side totals
    total_right: int = 0
    total_left: int = 0
    total_short: int = 0
    total_long: int = 0
    
    # Total misses analyzed
    total_misses: int = 0


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
    
    # NEW: Consistency metrics
    consistency: ConsistencyMetrics = field(default_factory=ConsistencyMetrics)
    
    # NEW: Tendency analysis
    tendencies: TendencyAnalysis = field(default_factory=TendencyAnalysis)
    
    # NEW: Miss distribution
    miss_distribution: MissDistribution = field(default_factory=MissDistribution)
    
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
        
        # Calculate consistency metrics
        stats.consistency = self._calculate_consistency()
        
        # Calculate tendency analysis
        stats.tendencies = self._calculate_tendencies()
        
        # Calculate miss distribution
        stats.miss_distribution = self._calculate_miss_distribution()
        
        return stats
    
    def _calculate_consistency(self, rolling_window: int = 10) -> ConsistencyMetrics:
        """Calculate consistency/variability metrics."""
        metrics = ConsistencyMetrics(rolling_window=rolling_window)
        
        if len(self.shots) < 3:
            return metrics
        
        # Extract data arrays
        speeds = np.array([s.speed_m_s for s in self.shots])
        directions = np.array([s.direction_deg for s in self.shots])
        distance_errors = np.array([s.distance_error_m for s in self.shots])
        
        # Standard deviations (full session)
        metrics.speed_stddev = float(np.std(speeds))
        metrics.direction_stddev = float(np.std(directions))
        metrics.distance_error_stddev = float(np.std(distance_errors))
        
        # Coefficient of variation for speed
        avg_speed = np.mean(speeds)
        if avg_speed > 0:
            metrics.speed_cv = float((metrics.speed_stddev / avg_speed) * 100)
        
        # Rolling consistency (last N shots)
        if len(self.shots) >= rolling_window:
            recent_shots = self.shots[-rolling_window:]
            recent_speeds = np.array([s.speed_m_s for s in recent_shots])
            recent_directions = np.array([s.direction_deg for s in recent_shots])
            metrics.rolling_speed_stddev = float(np.std(recent_speeds))
            metrics.rolling_direction_stddev = float(np.std(recent_directions))
        else:
            metrics.rolling_speed_stddev = metrics.speed_stddev
            metrics.rolling_direction_stddev = metrics.direction_stddev
        
        # Composite consistency score (0-100)
        # Lower stddev = higher score
        # Benchmarks: Tour pros have ~0.1 m/s speed stddev, ~1° direction stddev
        speed_score = max(0, 100 - (metrics.speed_stddev / 0.3) * 50)  # 0.3 m/s stddev = 50 score
        direction_score = max(0, 100 - (metrics.direction_stddev / 3.0) * 50)  # 3° stddev = 50 score
        distance_score = max(0, 100 - (metrics.distance_error_stddev / 0.5) * 50)  # 0.5m stddev = 50 score
        
        # Weighted composite (direction matters most for putting)
        metrics.consistency_score = float(
            direction_score * 0.4 +
            distance_score * 0.35 +
            speed_score * 0.25
        )
        metrics.consistency_score = min(100, max(0, metrics.consistency_score))
        
        return metrics
    
    def _calculate_tendencies(self) -> TendencyAnalysis:
        """Calculate tendency/bias analysis."""
        tendencies = TendencyAnalysis()
        
        if len(self.shots) < 3:
            return tendencies
        
        # Speed/distance bias
        distance_errors = [s.distance_error_m for s in self.shots]
        tendencies.distance_bias_m = float(np.mean(distance_errors))
        
        speeds = [s.speed_m_s for s in self.shots]
        tendencies.speed_bias_m_s = float(np.mean(speeds))  # Could compare to optimal
        
        # Direction bias
        directions = [s.direction_deg for s in self.shots]
        tendencies.direction_bias_deg = float(np.mean(directions))
        
        # Lateral bias (only from misses)
        misses = [s for s in self.shots if not s.is_made]
        if misses:
            lateral_misses = [s.lateral_miss_m for s in misses]
            tendencies.lateral_bias_m = float(np.mean(lateral_misses))
        
        # Determine dominant miss pattern from misses
        if len(misses) >= 3:
            quadrant_counts = {
                "right-short": 0,
                "right-long": 0,
                "left-short": 0,
                "left-long": 0
            }
            
            for miss in misses:
                is_right = miss.lateral_miss_m > 0
                is_long = miss.depth_miss_m > 0
                
                if is_right and not is_long:
                    quadrant_counts["right-short"] += 1
                elif is_right and is_long:
                    quadrant_counts["right-long"] += 1
                elif not is_right and not is_long:
                    quadrant_counts["left-short"] += 1
                else:
                    quadrant_counts["left-long"] += 1
            
            # Find dominant
            dominant = max(quadrant_counts, key=quadrant_counts.get)
            dominant_count = quadrant_counts[dominant]
            tendencies.dominant_miss = dominant
            tendencies.dominant_miss_percentage = float((dominant_count / len(misses)) * 100)
        
        # Generate tendency messages
        # Speed tendency
        if tendencies.distance_bias_m > 0.15:
            tendencies.speed_tendency = "hitting long"
        elif tendencies.distance_bias_m < -0.15:
            tendencies.speed_tendency = "hitting short"
        else:
            tendencies.speed_tendency = "neutral"
        
        # Direction tendency
        if tendencies.direction_bias_deg > 1.5:
            tendencies.direction_tendency = "pushing right"
        elif tendencies.direction_bias_deg < -1.5:
            tendencies.direction_tendency = "pulling left"
        else:
            tendencies.direction_tendency = "neutral"
        
        return tendencies
    
    def _calculate_miss_distribution(self) -> MissDistribution:
        """Calculate miss distribution by quadrant."""
        dist = MissDistribution()
        
        misses = [s for s in self.shots if not s.is_made]
        if not misses:
            return dist
        
        dist.total_misses = len(misses)
        
        for miss in misses:
            is_right = miss.lateral_miss_m > 0
            is_long = miss.depth_miss_m > 0
            
            if is_right and not is_long:
                dist.right_short += 1
            elif is_right and is_long:
                dist.right_long += 1
            elif not is_right and not is_long:
                dist.left_short += 1
            else:
                dist.left_long += 1
        
        # Calculate percentages
        total = dist.total_misses
        if total > 0:
            dist.right_short_pct = round((dist.right_short / total) * 100, 1)
            dist.right_long_pct = round((dist.right_long / total) * 100, 1)
            dist.left_short_pct = round((dist.left_short / total) * 100, 1)
            dist.left_long_pct = round((dist.left_long / total) * 100, 1)
        
        # Side totals
        dist.total_right = dist.right_short + dist.right_long
        dist.total_left = dist.left_short + dist.left_long
        dist.total_short = dist.right_short + dist.left_short
        dist.total_long = dist.right_long + dist.left_long
        
        return dist
    
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
            # NEW: Consistency metrics
            "consistency": {
                "speed_stddev": round(stats.consistency.speed_stddev, 3),
                "direction_stddev": round(stats.consistency.direction_stddev, 2),
                "distance_error_stddev": round(stats.consistency.distance_error_stddev, 3),
                "speed_cv": round(stats.consistency.speed_cv, 1),
                "consistency_score": round(stats.consistency.consistency_score, 1),
                "rolling_speed_stddev": round(stats.consistency.rolling_speed_stddev, 3),
                "rolling_direction_stddev": round(stats.consistency.rolling_direction_stddev, 2),
            },
            # NEW: Tendency analysis
            "tendencies": {
                "speed_bias_m_s": round(stats.tendencies.speed_bias_m_s, 3),
                "distance_bias_m": round(stats.tendencies.distance_bias_m, 3),
                "direction_bias_deg": round(stats.tendencies.direction_bias_deg, 2),
                "lateral_bias_m": round(stats.tendencies.lateral_bias_m, 3),
                "dominant_miss": stats.tendencies.dominant_miss,
                "dominant_miss_percentage": round(stats.tendencies.dominant_miss_percentage, 1),
                "speed_tendency": stats.tendencies.speed_tendency,
                "direction_tendency": stats.tendencies.direction_tendency,
            },
            # NEW: Miss distribution
            "miss_distribution": {
                "right_short": stats.miss_distribution.right_short,
                "right_long": stats.miss_distribution.right_long,
                "left_short": stats.miss_distribution.left_short,
                "left_long": stats.miss_distribution.left_long,
                "right_short_pct": stats.miss_distribution.right_short_pct,
                "right_long_pct": stats.miss_distribution.right_long_pct,
                "left_short_pct": stats.miss_distribution.left_short_pct,
                "left_long_pct": stats.miss_distribution.left_long_pct,
                "total_right": stats.miss_distribution.total_right,
                "total_left": stats.miss_distribution.total_left,
                "total_short": stats.miss_distribution.total_short,
                "total_long": stats.miss_distribution.total_long,
                "total_misses": stats.miss_distribution.total_misses,
            },
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
