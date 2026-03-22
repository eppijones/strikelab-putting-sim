#!/usr/bin/env python3
"""
StrikeLab Putting Sim - Single-command entry point.

Usage:
    python main.py              # Start with Arducam (default)
    python main.py --webcam     # Start with webcam
    python main.py --replay video.mp4  # Replay from video file
    python main.py --debug      # Enable debug logging
    python main.py --no-frontend  # Start backend only (skip frontend dev server)
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HEALTH_POLL_TIMEOUT_S = 60
HEALTH_POLL_INITIAL_DELAY_S = 0.5
HEALTH_POLL_MAX_DELAY_S = 3.0


def find_pid_on_port(port: int):
    if sys.platform != "win32":
        return None
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in parts:
                addr = parts[1]
                if addr.endswith(f":{port}"):
                    return int(parts[-1])
    except Exception:
        pass
    return None


def kill_port(port: int) -> bool:
    pid = find_pid_on_port(port)
    if pid is None:
        return False
    try:
        print(f"  Killing PID {pid} that was holding port {port}...")
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                capture_output=True, timeout=10,
            )
        else:
            os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        return True
    except Exception as e:
        print(f"  Could not kill PID {pid}: {e}")
        return False


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


def _start_backend(backend_args: list[str]) -> subprocess.Popen:
    """Launch backend as a subprocess so the launcher can poll health before starting frontend."""
    cmd = [sys.executable, "-m", "backend.main"] + backend_args
    return subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).parent),
        stdout=None,
        stderr=None,
    )


def _wait_for_backend_health(port: int, timeout_s: float = HEALTH_POLL_TIMEOUT_S, backend_proc: subprocess.Popen | None = None) -> bool:
    """
    Poll /api/health with exponential back-off.
    Returns True once the backend reports status=ready or status=degraded
    (both mean cameras are initialised or failed gracefully).
    """
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout_s
    delay = HEALTH_POLL_INITIAL_DELAY_S

    while time.time() < deadline:
        if backend_proc and backend_proc.poll() is not None:
            print(f"\n*** Backend process exited unexpectedly (code {backend_proc.returncode}) ***")
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                status = data.get("status", "")
                phase = data.get("startup_phase", "")
                cam_ready = data.get("camera_ready", False)

                if status in ("ready", "degraded"):
                    arducam = data.get("arducam") or {}
                    print(
                        f"  Backend ready: status={status}, camera_ready={cam_ready}, "
                        f"arducam_fps={arducam.get('fps', 0)}, "
                        f"sustained_startup_fps={arducam.get('sustained_startup_fps', 0)}, "
                        f"resolution={arducam.get('resolution', 'unknown')}"
                    )
                    if data.get("degraded_reasons"):
                        print(f"  Degraded reasons: {data['degraded_reasons']}")
                    return True

                print(f"  Backend starting: phase={phase}, status={status}...")
        except (urllib.error.URLError, OSError, ValueError):
            pass

        time.sleep(delay)
        delay = min(delay * 1.5, HEALTH_POLL_MAX_DELAY_S)

    print("\n*** Backend health check timed out ***")
    return False


def _start_frontend(frontend_dir: Path) -> subprocess.Popen | None:
    try:
        npm_cmd = "npm"
        env = os.environ.copy()
        if sys.platform == "win32":
            npm_cmd = "npm.cmd"
            fnm_node = Path(os.environ.get("APPDATA", "")) / "fnm" / "node-versions"
            if fnm_node.exists():
                versions = sorted(fnm_node.iterdir(), reverse=True)
                for v in versions:
                    install_dir = v / "installation"
                    if (install_dir / "npm.cmd").exists():
                        env["PATH"] = str(install_dir) + ";" + env.get("PATH", "")
                        break

        proc = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=str(frontend_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        time.sleep(2)
        if proc.poll() is not None:
            print("Warning: Frontend failed to start. Continuing with backend only.")
            return None
        print("Frontend dev server started (check output for URL)")
        return proc
    except FileNotFoundError:
        print("Warning: npm not found. Skipping frontend. Install Node.js or use --no-frontend")
    except Exception as e:
        print(f"Warning: Could not start frontend: {e}")
    return None


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

    # Auto-kill previous instance if port is in use
    if is_port_in_use(args.port):
        print(f"\nPort {args.port} is in use — stopping previous instance...")
        if kill_port(args.port):
            time.sleep(1)
            if is_port_in_use(args.port):
                print(f"*** ERROR: Port {args.port} is STILL in use after kill. ***")
                print("Close the other app manually and try again.")
                sys.exit(1)
            print(f"  Port {args.port} is now free.\n")
        else:
            print(f"*** ERROR: Could not free port {args.port}. ***")
            print("Close the other app manually and try again.")
            sys.exit(1)

    # Build backend arguments
    backend_args: list[str] = []
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

    frontend_process: subprocess.Popen | None = None
    backend_process: subprocess.Popen | None = None
    frontend_dir = Path(__file__).parent / "frontend"

    def cleanup(signum=None, frame=None):
        if frontend_process:
            print("\nStopping frontend dev server...")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        if backend_process:
            print("Stopping backend...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # --- Step 1: Start backend ---
        print(f"\nStarting backend on http://{args.host}:{args.port}")
        backend_process = _start_backend(backend_args)

        # --- Step 2: Wait for backend to become healthy ---
        print("Waiting for backend readiness...")
        if not _wait_for_backend_health(args.port, backend_proc=backend_process):
            print(
                "\n*** Backend did not become healthy within timeout. ***\n"
                "Check camera connections, USB bandwidth, and backend logs above."
            )
            cleanup()

        # --- Step 3: Start frontend only AFTER backend is ready ---
        if not args.no_frontend and frontend_dir.exists():
            print("Starting frontend dev server...")
            frontend_process = _start_frontend(frontend_dir)

        print("\nPress Ctrl+C to stop both servers\n")

        # Block on backend process
        backend_process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
