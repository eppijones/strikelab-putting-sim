#!/usr/bin/env python3
"""
StrikeLab Putting Sim - Simple entry point.

Usage:
    python main.py              # Start with Arducam (default)
    python main.py --webcam     # Start with webcam
    python main.py --replay video.mp4  # Replay from video file
    python main.py --debug      # Enable debug logging
    python main.py --no-frontend  # Start backend only (skip frontend dev server)
"""

import argparse
import subprocess
import sys
import os
import signal
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="StrikeLab Putting Sim")
    parser.add_argument("--arducam", action="store_true", help="Use Arducam OV9281")
    parser.add_argument("--webcam", action="store_true", help="Use standard webcam")
    parser.add_argument("--replay", type=str, help="Replay from video file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-frontend", action="store_true", help="Skip starting frontend dev server")
    
    args = parser.parse_args()
    
    frontend_process = None
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Start frontend dev server if not disabled
    if not args.no_frontend and frontend_dir.exists():
        print("Starting frontend dev server...")
        try:
            # Check if npm is available
            npm_cmd = "npm"
            if sys.platform == "win32":
                npm_cmd = "npm.cmd"
            
            # Start frontend in background
            frontend_process = subprocess.Popen(
                [npm_cmd, "run", "dev"],
                cwd=str(frontend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Wait a moment for the frontend to start
            time.sleep(2)
            
            # Check if it started successfully
            if frontend_process.poll() is not None:
                print("Warning: Frontend failed to start. Continuing with backend only.")
                frontend_process = None
            else:
                print("Frontend dev server started (check output for URL)")
                
        except FileNotFoundError:
            print("Warning: npm not found. Skipping frontend. Install Node.js or use --no-frontend")
        except Exception as e:
            print(f"Warning: Could not start frontend: {e}")
    
    # Build backend arguments (pass through all relevant args)
    backend_args = []
    if args.arducam:
        backend_args.append("--arducam")
    if args.webcam:
        backend_args.append("--webcam")
    if args.replay:
        backend_args.extend(["--replay", args.replay])
    if args.host != "0.0.0.0":
        backend_args.extend(["--host", args.host])
    if args.port != 8000:
        backend_args.extend(["--port", str(args.port)])
    if args.debug:
        backend_args.append("--debug")
    
    def cleanup(signum=None, frame=None):
        """Clean up child processes on exit."""
        if frontend_process:
            print("\nStopping frontend dev server...")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Import and run backend
        print(f"\nStarting backend on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop both servers\n")
        
        # Modify sys.argv to pass our parsed args to backend
        sys.argv = ["backend.main"] + backend_args
        
        from backend.main import main as backend_main
        backend_main()
        
    finally:
        cleanup()


if __name__ == "__main__":
    main()
