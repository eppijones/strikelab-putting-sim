"""
SQLite database for persistent shot history storage.
Stores all shots across sessions for long-term analytics.
"""

import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from .game_logic import ShotResult

logger = logging.getLogger(__name__)

# Database file location
DB_PATH = Path(__file__).parent.parent / "data" / "putting_history.db"


@dataclass
class ShotRecord:
    """A single shot record from the database."""
    id: int
    timestamp: str
    session_id: str
    speed_m_s: float
    distance_m: float
    direction_deg: float
    target_distance_m: float
    result: str
    is_made: bool
    distance_to_hole_m: float
    lateral_miss_m: float
    depth_miss_m: float


class PuttingDatabase:
    """
    SQLite database for storing putting history.
    Provides persistence across app restarts.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the database.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/putting_history.db
        """
        self.db_path = db_path or DB_PATH
        self._ensure_directory()
        self._init_db()
    
    def _ensure_directory(self):
        """Ensure the data directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Shots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    speed_m_s REAL NOT NULL,
                    distance_m REAL NOT NULL,
                    direction_deg REAL NOT NULL,
                    target_distance_m REAL NOT NULL,
                    result TEXT NOT NULL,
                    is_made INTEGER NOT NULL,
                    distance_to_hole_m REAL,
                    lateral_miss_m REAL,
                    depth_miss_m REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_putts INTEGER DEFAULT 0,
                    putts_made INTEGER DEFAULT 0,
                    best_streak INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)
            
            # Index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_shots_session 
                ON shots(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_shots_timestamp 
                ON shots(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_shots_result 
                ON shots(result)
            """)
            
            logger.info(f"Database initialized at {self.db_path}")
    
    def save_shot(
        self,
        session_id: str,
        speed_m_s: float,
        distance_m: float,
        direction_deg: float,
        target_distance_m: float,
        result: ShotResult,
        distance_to_hole_m: float,
        lateral_miss_m: float,
        depth_miss_m: float
    ) -> int:
        """
        Save a shot to the database.
        
        Returns:
            The ID of the inserted shot
        """
        timestamp = datetime.utcnow().isoformat()
        is_made = 1 if result == ShotResult.MADE else 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO shots (
                    timestamp, session_id, speed_m_s, distance_m, direction_deg,
                    target_distance_m, result, is_made, distance_to_hole_m,
                    lateral_miss_m, depth_miss_m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, session_id, speed_m_s, distance_m, direction_deg,
                target_distance_m, result.value, is_made, distance_to_hole_m,
                lateral_miss_m, depth_miss_m
            ))
            
            shot_id = cursor.lastrowid
            logger.info(f"Saved shot {shot_id}: {result.value}, distance={distance_m:.2f}m")
            return shot_id
    
    def start_session(self, session_id: str) -> None:
        """Record a new session start."""
        timestamp = datetime.utcnow().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (id, start_time)
                VALUES (?, ?)
            """, (session_id, timestamp))
    
    def end_session(
        self,
        session_id: str,
        total_putts: int,
        putts_made: int,
        best_streak: int
    ) -> None:
        """Update session with final stats."""
        timestamp = datetime.utcnow().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET end_time = ?, total_putts = ?, putts_made = ?, best_streak = ?
                WHERE id = ?
            """, (timestamp, total_putts, putts_made, best_streak, session_id))
    
    def get_shots(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ShotRecord]:
        """
        Get shots from the database.
        
        Args:
            session_id: Filter by session ID (optional)
            limit: Maximum number of shots to return
            offset: Number of shots to skip
        
        Returns:
            List of ShotRecord objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT * FROM shots 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (session_id, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM shots 
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            rows = cursor.fetchall()
            return [ShotRecord(**dict(row)) for row in rows]
    
    def get_stats(
        self,
        days: int = 30,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics.
        
        Args:
            days: Number of days to include (0 = all time)
            session_id: Filter by session ID (optional)
        
        Returns:
            Dictionary of statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build date filter
            date_filter = ""
            params: List[Any] = []
            
            if days > 0:
                date_filter = "WHERE timestamp >= datetime('now', ?)"
                params.append(f'-{days} days')
            
            if session_id:
                if date_filter:
                    date_filter += " AND session_id = ?"
                else:
                    date_filter = "WHERE session_id = ?"
                params.append(session_id)
            
            # Total stats
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_putts,
                    SUM(is_made) as putts_made,
                    AVG(speed_m_s) as avg_speed,
                    AVG(distance_m) as avg_distance,
                    AVG(ABS(direction_deg)) as avg_line_error
                FROM shots
                {date_filter}
            """, params)
            
            row = cursor.fetchone()
            total_putts = row['total_putts'] or 0
            putts_made = row['putts_made'] or 0
            
            # Stats by distance band
            cursor.execute(f"""
                SELECT 
                    CASE 
                        WHEN target_distance_m < 1 THEN '0-1m'
                        WHEN target_distance_m < 2 THEN '1-2m'
                        WHEN target_distance_m < 3 THEN '2-3m'
                        WHEN target_distance_m < 4 THEN '3-4m'
                        WHEN target_distance_m < 5 THEN '4-5m'
                        WHEN target_distance_m < 6 THEN '5-6m'
                        ELSE '6m+'
                    END as distance_band,
                    COUNT(*) as total,
                    SUM(is_made) as made,
                    ROUND(100.0 * SUM(is_made) / COUNT(*), 1) as percentage
                FROM shots
                {date_filter}
                GROUP BY distance_band
                ORDER BY distance_band
            """, params)
            
            by_distance = {
                row['distance_band']: {
                    'total': row['total'],
                    'made': row['made'],
                    'percentage': row['percentage']
                }
                for row in cursor.fetchall()
            }
            
            # Stats by result type
            cursor.execute(f"""
                SELECT 
                    result,
                    COUNT(*) as count,
                    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM shots {date_filter}), 1) as percentage
                FROM shots
                {date_filter}
                GROUP BY result
            """, params + params)  # Need params twice for subquery
            
            by_result = {
                row['result']: {
                    'count': row['count'],
                    'percentage': row['percentage']
                }
                for row in cursor.fetchall()
            }
            
            return {
                'total_putts': total_putts,
                'putts_made': putts_made,
                'make_percentage': round((putts_made / total_putts * 100) if total_putts > 0 else 0, 1),
                'avg_speed_m_s': round(row['avg_speed'] or 0, 2),
                'avg_distance_m': round(row['avg_distance'] or 0, 2),
                'avg_line_error_deg': round(row['avg_line_error'] or 0, 2),
                'by_distance': by_distance,
                'by_result': by_result,
                'period_days': days
            }
    
    def get_recent_trend(self, n_shots: int = 20) -> Dict[str, Any]:
        """
        Get trend data for the most recent N shots.
        
        Returns:
            Dictionary with trend data for charting
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    timestamp,
                    speed_m_s,
                    distance_m,
                    direction_deg,
                    target_distance_m,
                    result,
                    is_made,
                    distance_to_hole_m
                FROM shots
                ORDER BY timestamp DESC
                LIMIT ?
            """, (n_shots,))
            
            rows = cursor.fetchall()
            
            # Reverse to get chronological order
            rows = list(reversed(rows))
            
            return {
                'timestamps': [row['timestamp'] for row in rows],
                'speeds': [row['speed_m_s'] for row in rows],
                'distances': [row['distance_m'] for row in rows],
                'directions': [row['direction_deg'] for row in rows],
                'results': [row['result'] for row in rows],
                'made': [bool(row['is_made']) for row in rows],
                'distance_to_hole': [row['distance_to_hole_m'] for row in rows]
            }
    
    def export_csv(self, filepath: Path, session_id: Optional[str] = None) -> int:
        """
        Export shots to CSV file.
        
        Args:
            filepath: Path to output CSV file
            session_id: Optional session filter
        
        Returns:
            Number of shots exported
        """
        import csv
        
        shots = self.get_shots(session_id=session_id, limit=10000)
        
        with open(filepath, 'w', newline='') as f:
            if not shots:
                return 0
            
            writer = csv.DictWriter(f, fieldnames=asdict(shots[0]).keys())
            writer.writeheader()
            for shot in shots:
                writer.writerow(asdict(shot))
        
        logger.info(f"Exported {len(shots)} shots to {filepath}")
        return len(shots)
    
    def clear_all(self) -> None:
        """Clear all data from the database. Use with caution!"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM shots")
            cursor.execute("DELETE FROM sessions")
            logger.warning("All database data cleared")


# Singleton instance
_database: Optional[PuttingDatabase] = None


def get_database() -> PuttingDatabase:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = PuttingDatabase()
    return _database
