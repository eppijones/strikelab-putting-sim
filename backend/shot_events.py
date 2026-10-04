"""Durable, compact launch delivery. Camera images never enter this journal."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class ShotJournal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS launches (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    shot_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS consumers (
                    client_id TEXT PRIMARY KEY, ack_seq INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS identity (device_id TEXT NOT NULL);
            ''')
            row = db.execute('SELECT device_id FROM identity').fetchone()
            self.device_id = row[0] if row else str(uuid.uuid4())
            if not row:
                db.execute('INSERT INTO identity VALUES (?)', (self.device_id,))
        self.session_id = str(uuid.uuid4())

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def append(self, *, shot_id: str, speed: float, direction: float,
               forward: float, ppm: float, calibration_source: str,
               calibrated: bool, estimated: bool = False) -> dict:
        if not all(math.isfinite(v) for v in (speed, direction, forward, ppm)):
            raise ValueError('Launch measurements must be finite')
        if not 0 < speed <= 100 or ppm <= 0:
            raise ValueError('Launch speed or calibration is out of range')
        calibration = json.dumps([round(ppm, 6), forward, calibration_source])
        event = {
            'type': 'shot.launched', 'version': 1, 'shot_id': shot_id,
            'device_id': self.device_id, 'session_id': self.session_id,
            'timestamp_ms': time.time_ns() // 1_000_000,
            'calibration_id': hashlib.sha256(calibration.encode()).hexdigest()[:24],
            'speed_m_s': speed, 'direction_deg': (direction - forward + 180) % 360 - 180,
            'direction_frame': 'target-relative-positive-right',
            'launch_angle_deg': None, 'spin_rpm': None,
            'quality': {
                'speed': 'unavailable' if not calibrated else ('estimated' if estimated else 'measured'),
                'direction': 'measured' if calibrated else 'unavailable',
                'launch': 'unavailable', 'spin': 'unavailable',
            },
        }
        with self.lock, self.connect() as db:
            db.execute('INSERT OR IGNORE INTO launches(shot_id,payload) VALUES (?,?)',
                       (shot_id, json.dumps(event, allow_nan=False)))
            row = db.execute('SELECT seq,payload FROM launches WHERE shot_id=?', (shot_id,)).fetchone()
        return {'seq': row[0], **json.loads(row[1])}

    def head(self) -> int:
        with self.connect() as db:
            return db.execute('SELECT COALESCE(MAX(seq),0) FROM launches').fetchone()[0]

    def read(self, after: int, limit: int = 32) -> list[dict]:
        with self.connect() as db:
            rows = db.execute('SELECT seq,payload FROM launches WHERE seq>? ORDER BY seq LIMIT ?',
                              (after, min(limit, 128))).fetchall()
        return [{'seq': seq, **json.loads(payload)} for seq, payload in rows]

    def acknowledge(self, client_id: str, seq: int):
        if seq < 0 or seq > self.head():
            raise ValueError('Invalid acknowledgement')
        with self.connect() as db:
            db.execute('INSERT INTO consumers VALUES (?,?) ON CONFLICT(client_id) DO UPDATE '
                       'SET ack_seq=MAX(ack_seq,excluded.ack_seq)', (client_id, seq))


_journal = None
_journal_lock = threading.Lock()


def get_shot_journal() -> ShotJournal:
    global _journal
    with _journal_lock:
        if _journal is None:
            _journal = ShotJournal(Path(__file__).resolve().parents[1] / 'data' / 'shot_events.db')
        return _journal
