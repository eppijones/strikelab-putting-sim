#!/usr/bin/env python3
"""
StrikeLab Putting Sim - Simple entry point.

Usage:
    python main.py              # Start with Arducam (default)
    python main.py --webcam     # Start with webcam
    python main.py --replay video.mp4  # Replay from video file
    python main.py --debug      # Enable debug logging
"""

from backend.main import main

if __name__ == "__main__":
    main()
