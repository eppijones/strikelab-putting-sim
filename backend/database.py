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
class UserRecord:
    """A single user record."""
    id: int
    name: str
    handicap: float
    created_at: str


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
    user_id: Optional[int] = None
    created_at: Optional[str] = None


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
        self._check_migrations()
        self._create_indexes()
    
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
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    handicap REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

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
                    user_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
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
                    notes TEXT,
                    user_id INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            logger.info(f"Database initialized at {self.db_path}")
    
    def _create_indexes(self):
        """Create database indexes after migrations have run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
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
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_shots_user 
                ON shots(user_id)
            """)

    def _check_migrations(self):
        """Check and apply necessary migrations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if user_id column exists in shots table
            cursor.execute("PRAGMA table_info(shots)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'user_id' not in columns:
                logger.info("Migrating shots table: adding user_id column")
                cursor.execute("ALTER TABLE shots ADD COLUMN user_id INTEGER REFERENCES users(id)")
            
            # Check if user_id column exists in sessions table
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'user_id' not in columns:
                logger.info("Migrating sessions table: adding user_id column")
                cursor.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users(id)")

    # --- User Management ---

    def create_user(self, name: str, handicap: float = 0.0) -> int:
        """Create a new user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (name, handicap)
                VALUES (?, ?)
            """, (name, handicap))
            return cursor.lastrowid

    def get_users(self) -> List[UserRecord]:
        """Get all users."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY name")
            rows = cursor.fetchall()
            return [UserRecord(**dict(row)) for row in rows]

    def delete_user(self, user_id: int) -> None:
        """Delete a user and their data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Optional: Decide if we want to cascade delete or keep shots as anonymous
            # For now, let's keep shots but nullify user_id, or we can delete.
            # Let's delete for privacy/cleanup.
            cursor.execute("DELETE FROM shots WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def reset_user_data(self, user_id: int) -> Dict[str, int]:
        """
        Reset all data for a user without deleting the user account.
        Clears all shots and sessions for this user.
        
        Args:
            user_id: The user ID to reset data for
            
        Returns:
            Dictionary with counts of deleted records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count records before deletion
            cursor.execute("SELECT COUNT(*) FROM shots WHERE user_id = ?", (user_id,))
            shots_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,))
            sessions_count = cursor.fetchone()[0]
            
            # Delete the data
            cursor.execute("DELETE FROM shots WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            
            logger.info(f"Reset data for user {user_id}: deleted {shots_count} shots, {sessions_count} sessions")
            
            return {
                "shots_deleted": shots_count,
                "sessions_deleted": sessions_count
            }

    def delete_shot(self, shot_id: int) -> bool:
        """Delete a specific shot."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM shots WHERE id = ?", (shot_id,))
            return cursor.rowcount > 0

    # --- Shot Management ---

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
        depth_miss_m: float,
        user_id: Optional[int] = None
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
                    lateral_miss_m, depth_miss_m, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, session_id, speed_m_s, distance_m, direction_deg,
                target_distance_m, result.value, is_made, distance_to_hole_m,
                lateral_miss_m, depth_miss_m, user_id
            ))
            
            shot_id = cursor.lastrowid
            logger.info(f"Saved shot {shot_id}: {result.value}, distance={distance_m:.2f}m, user_id={user_id}")
            return shot_id
    
    def start_session(self, session_id: str, user_id: Optional[int] = None) -> None:
        """Record a new session start."""
        timestamp = datetime.utcnow().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (id, start_time, user_id)
                VALUES (?, ?, ?)
            """, (session_id, timestamp, user_id))
    
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
        user_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ShotRecord]:
        """
        Get shots from the database.
        
        Args:
            session_id: Filter by session ID (optional)
            user_id: Filter by user ID (optional)
            limit: Maximum number of shots to return
            offset: Number of shots to skip
        
        Returns:
            List of ShotRecord objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM shots WHERE 1=1"
            params = []
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
                
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            return [ShotRecord(**dict(row)) for row in rows]
    
    def get_stats(
        self,
        days: int = 30,
        session_id: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics.
        
        Args:
            days: Number of days to include (0 = all time)
            session_id: Filter by session ID (optional)
            user_id: Filter by user ID (optional)
        
        Returns:
            Dictionary of statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build date filter
            where_clauses = []
            params: List[Any] = []
            
            if days > 0:
                where_clauses.append("timestamp >= datetime('now', ?)")
                params.append(f'-{days} days')
            
            if session_id:
                where_clauses.append("session_id = ?")
                params.append(session_id)
                
            if user_id:
                where_clauses.append("user_id = ?")
                params.append(user_id)
            
            where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            # Total stats
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_putts,
                    SUM(is_made) as putts_made,
                    AVG(speed_m_s) as avg_speed,
                    AVG(distance_m) as avg_distance,
                    AVG(ABS(direction_deg)) as avg_line_error
                FROM shots
                {where_sql}
            """, params)
            
            stats_row = cursor.fetchone()
            total_putts = stats_row['total_putts'] or 0
            putts_made = stats_row['putts_made'] or 0
            avg_speed = stats_row['avg_speed'] or 0
            avg_distance = stats_row['avg_distance'] or 0
            avg_line_error = stats_row['avg_line_error'] or 0
            
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
                {where_sql}
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
            # Need to re-supply params for the subquery in the percentage calculation
            # Or just calculate percentage in python to avoid complex SQL param duplication
            cursor.execute(f"""
                SELECT 
                    result,
                    COUNT(*) as count
                FROM shots
                {where_sql}
                GROUP BY result
            """, params)
            
            result_rows = cursor.fetchall()
            by_result = {}
            for row in result_rows:
                count = row['count']
                percentage = round((count / total_putts * 100), 1) if total_putts > 0 else 0.0
                by_result[row['result']] = {
                    'count': count,
                    'percentage': percentage
                }
            
            return {
                'total_putts': total_putts,
                'putts_made': putts_made,
                'make_percentage': round((putts_made / total_putts * 100) if total_putts > 0 else 0, 1),
                'avg_speed_m_s': round(avg_speed, 2),
                'avg_distance_m': round(avg_distance, 2),
                'avg_line_error_deg': round(avg_line_error, 2),
                'by_distance': by_distance,
                'by_result': by_result,
                'period_days': days
            }
    
    def get_recent_trend(self, n_shots: int = 20, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get trend data for the most recent N shots.
        
        Returns:
            Dictionary with trend data for charting
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
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
            """
            params = []
            
            if user_id:
                query += " WHERE user_id = ?"
                params.append(user_id)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(n_shots)
            
            cursor.execute(query, params)
            
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
    
    def export_csv(self, filepath: Path, session_id: Optional[str] = None, user_id: Optional[int] = None) -> int:
        """
        Export shots to CSV file.
        
        Args:
            filepath: Path to output CSV file
            session_id: Optional session filter
            user_id: Optional user filter
        
        Returns:
            Number of shots exported
        """
        import csv
        
        shots = self.get_shots(session_id=session_id, user_id=user_id, limit=10000)
        
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
            cursor.execute("DELETE FROM users")
            logger.warning("All database data cleared")


# Singleton instance
_database: Optional[PuttingDatabase] = None


def get_database() -> PuttingDatabase:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = PuttingDatabase()
    return _database
