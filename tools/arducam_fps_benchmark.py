#!/usr/bin/env python3
"""
Arducam OV9281 120 FPS Benchmark Tool
======================================
Isolated test to verify stable 120 fps capture from the Arducam OV9281.

Runs multiple test phases, each stripping away layers of overhead to find
exactly where frames are lost:

  Phase 1 - Raw capture baseline (tight loop, no processing)
  Phase 2 - Backend comparison (DSHOW vs MSMF vs CAP_ANY)
  Phase 3 - FOURCC / pixel format (Y800, GREY, MJPG, YUY2)
  Phase 4 - Exposure & auto-exposure sweeps
  Phase 5 - CAP_PROP_CONVERT_RGB disabled (avoid internal BGR conversion)
  Phase 6 - Buffer size variations
  Phase 7 - Sustained run (10+ seconds with per-frame jitter analysis)

Usage:
    python tools/arducam_fps_benchmark.py                   # Full benchmark
    python tools/arducam_fps_benchmark.py --device 1        # Specific device
    python tools/arducam_fps_benchmark.py --phase 7         # Run only phase 7
    python tools/arducam_fps_benchmark.py --scan             # Scan all devices first
    python tools/arducam_fps_benchmark.py --duration 15     # Sustained test duration
"""

import argparse
import csv
import ctypes
import io
import os
import sys
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np

# -- Constants ----------------------------------------------------------
ARDUCAM_WIDTH = 1280
ARDUCAM_HEIGHT = 800
TARGET_FPS = 120

WARMUP_FRAMES = 60
DEFAULT_SUSTAINED_DURATION_S = 10.0


# -- Helpers ------------------------------------------------------------
@dataclass
class TimingResult:
    label: str
    total_frames: int = 0
    elapsed_s: float = 0.0
    fps: float = 0.0
    dt_mean_ms: float = 0.0
    dt_median_ms: float = 0.0
    dt_std_ms: float = 0.0
    dt_p1_ms: float = 0.0
    dt_p99_ms: float = 0.0
    dt_min_ms: float = 0.0
    dt_max_ms: float = 0.0
    jitter_ms: float = 0.0   # std of (dt - ideal_dt)
    dropped_estimate: int = 0
    resolution: Tuple[int, int] = (0, 0)
    fourcc: str = ""
    backend: str = ""
    extra: str = ""

    def summary_line(self) -> str:
        status = "PASS" if self.fps >= 115.0 else ("WARN" if self.fps >= 90 else "FAIL")
        return (
            f"[{status}] {self.label:42s} | "
            f"{self.fps:6.1f} fps | "
            f"dt={self.dt_mean_ms:5.2f}±{self.dt_std_ms:4.2f}ms | "
            f"p1/p99={self.dt_p1_ms:5.2f}/{self.dt_p99_ms:5.2f}ms | "
            f"jitter={self.jitter_ms:4.2f}ms | "
            f"{self.resolution[0]}x{self.resolution[1]} {self.fourcc} {self.backend}"
        )


def fourcc_int(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code) if len(code) == 4 else 0


def fourcc_to_str(v: int) -> str:
    try:
        return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))
    except Exception:
        return "????"


def compute_timing(label: str, dts_ns: List[int], res: Tuple[int, int],
                   fourcc: str, backend: str, extra: str = "") -> TimingResult:
    if not dts_ns:
        return TimingResult(label=label)
    dts_ms = [dt / 1e6 for dt in dts_ns]
    ideal_dt_ms = 1000.0 / TARGET_FPS
    jitters_ms = [dt - ideal_dt_ms for dt in dts_ms]
    elapsed_s = sum(dts_ms) / 1000.0
    total_frames = len(dts_ms) + 1

    dropped = sum(1 for dt in dts_ms if dt > ideal_dt_ms * 1.8)

    return TimingResult(
        label=label,
        total_frames=total_frames,
        elapsed_s=elapsed_s,
        fps=len(dts_ms) / elapsed_s if elapsed_s > 0 else 0,
        dt_mean_ms=statistics.mean(dts_ms),
        dt_median_ms=statistics.median(dts_ms),
        dt_std_ms=statistics.stdev(dts_ms) if len(dts_ms) > 1 else 0,
        dt_p1_ms=float(np.percentile(dts_ms, 1)),
        dt_p99_ms=float(np.percentile(dts_ms, 99)),
        dt_min_ms=min(dts_ms),
        dt_max_ms=max(dts_ms),
        jitter_ms=statistics.stdev(jitters_ms) if len(jitters_ms) > 1 else 0,
        dropped_estimate=dropped,
        resolution=res,
        fourcc=fourcc,
        backend=backend,
        extra=extra,
    )


def set_high_process_priority():
    """Elevate process priority to reduce OS scheduling jitter."""
    try:
        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetCurrentProcess()
            HIGH_PRIORITY_CLASS = 0x00000080
            kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
            print("  Process priority set to HIGH")
        else:
            os.nice(-10)
            print("  Process nice set to -10")
    except Exception as e:
        print(f"  Could not set high priority: {e}")


# -- Device scanner -----------------------------------------------------
def scan_devices(max_devices: int = 10) -> List[Dict]:
    """Probe all device indices and report resolution/FPS."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    results = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ARDUCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARDUCAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_reported = cap.get(cv2.CAP_PROP_FPS)
        fc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
        ret, frame = cap.read()
        actual_res = (frame.shape[1], frame.shape[0]) if ret and frame is not None else (w, h)
        cap.release()
        aspect = actual_res[0] / actual_res[1] if actual_res[1] > 0 else 0
        tag = ""
        if aspect >= 2.5:
            tag = "stereo-SBS"
        elif abs(actual_res[0] - 1280) <= 64 and abs(actual_res[1] - 800) <= 64:
            tag = "ARDUCAM-candidate"
        results.append({
            "idx": idx, "res": actual_res, "reported_fps": fps_reported,
            "fourcc": fc, "tag": tag,
        })
        print(f"  Device {idx}: {actual_res[0]}x{actual_res[1]} @ {fps_reported:.0f}fps "
              f"fourcc={fc}  {tag}")
    return results


# -- Core benchmark function --------------------------------------------
def benchmark_capture(
    device_id: int,
    backend: int,
    backend_name: str,
    label: str,
    duration_s: float = 3.0,
    warmup_frames: int = WARMUP_FRAMES,
    fourcc_code: Optional[str] = None,
    set_exposure: Optional[Tuple[float, float]] = None,  # (auto_exposure, exposure)
    convert_rgb: Optional[bool] = None,
    buffer_size: Optional[int] = None,
    width: int = ARDUCAM_WIDTH,
    height: int = ARDUCAM_HEIGHT,
    target_fps: int = TARGET_FPS,
) -> Optional[TimingResult]:
    """
    Open camera, configure, warmup, then measure frame timing for `duration_s`.
    Returns TimingResult or None if open failed.
    """
    cap = cv2.VideoCapture(device_id, backend)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    if fourcc_code:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_int(fourcc_code))

    if convert_rgb is not None:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0 if not convert_rgb else 1)

    if buffer_size is not None:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        except Exception:
            pass

    if set_exposure is not None:
        ae, ev = set_exposure
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae)
        cap.set(cv2.CAP_PROP_EXPOSURE, ev)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # Reject stereo SBS cameras
    if actual_h > 0 and (actual_w / actual_h) >= 2.5:
        cap.release()
        return None

    # Reject wrong resolution
    if abs(actual_w - ARDUCAM_WIDTH) > 64 or abs(actual_h - ARDUCAM_HEIGHT) > 64:
        cap.release()
        return None

    # Warmup: drain initial frames so driver/sensor settles
    for _ in range(warmup_frames):
        cap.read()

    # Measurement: tight loop collecting monotonic timestamps per frame
    timestamps_ns: List[int] = []
    end_time = time.perf_counter() + duration_s

    while time.perf_counter() < end_time:
        ret, frame = cap.read()
        if ret and frame is not None:
            timestamps_ns.append(time.perf_counter_ns())

    cap.release()

    if len(timestamps_ns) < 2:
        return TimingResult(label=label, resolution=(actual_w, actual_h),
                            fourcc=actual_fourcc, backend=backend_name)

    dts_ns = [timestamps_ns[i] - timestamps_ns[i - 1] for i in range(1, len(timestamps_ns))]
    extra = f"reported={actual_fps:.0f}fps"
    return compute_timing(label, dts_ns, (actual_w, actual_h),
                          actual_fourcc, backend_name, extra)


# -- Phases -------------------------------------------------------------

def phase1_raw_baseline(device_id: int) -> List[TimingResult]:
    """Phase 1: Raw capture baseline - DSHOW, default settings."""
    print("\n== Phase 1: Raw capture baseline (DSHOW, defaults) ==")
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    bname = "DSHOW" if sys.platform == "win32" else "ANY"
    r = benchmark_capture(device_id, backend, bname, "P1: Raw baseline (defaults)", duration_s=3.0)
    results = [r] if r else []
    for r in results:
        print(f"  {r.summary_line()}")
    return results


def phase2_backend_comparison(device_id: int) -> List[TimingResult]:
    """Phase 2: Compare capture backends."""
    print("\n== Phase 2: Backend comparison ==")
    backends = []
    if sys.platform == "win32":
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "ANY"),
        ]
    else:
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "ANY"),
        ]

    results = []
    for be, name in backends:
        print(f"  Testing {name}...", end=" ", flush=True)
        r = benchmark_capture(device_id, be, name, f"P2: {name} backend", duration_s=3.0)
        if r:
            print(f"{r.fps:.1f} fps")
            results.append(r)
        else:
            print("FAILED (could not open or wrong resolution)")
        time.sleep(0.5)  # let driver fully release

    for r in results:
        print(f"  {r.summary_line()}")
    return results


def phase3_fourcc_formats(device_id: int, backend: int, backend_name: str) -> List[TimingResult]:
    """Phase 3: Try different pixel formats."""
    print(f"\n== Phase 3: FOURCC / pixel format (backend={backend_name}) ==")
    formats = ["Y800", "GREY", "MJPG", "YUY2", "NV12", "I420"]

    results = []
    for fc in formats:
        print(f"  Testing FOURCC={fc}...", end=" ", flush=True)
        r = benchmark_capture(device_id, backend, backend_name,
                              f"P3: FOURCC={fc}", duration_s=3.0, fourcc_code=fc)
        if r:
            print(f"{r.fps:.1f} fps (actual fourcc={r.fourcc})")
            results.append(r)
        else:
            print("FAILED")
        time.sleep(0.3)

    # Also test without setting fourcc (driver default)
    print(f"  Testing FOURCC=<default>...", end=" ", flush=True)
    r = benchmark_capture(device_id, backend, backend_name,
                          f"P3: FOURCC=<default>", duration_s=3.0)
    if r:
        print(f"{r.fps:.1f} fps (actual fourcc={r.fourcc})")
        results.append(r)
    else:
        print("FAILED")

    for r in results:
        print(f"  {r.summary_line()}")
    return results


def phase4_exposure_sweep(device_id: int, backend: int, backend_name: str,
                          fourcc_code: Optional[str] = None) -> List[TimingResult]:
    """Phase 4: Exposure and auto-exposure combinations."""
    print(f"\n== Phase 4: Exposure sweep (backend={backend_name}, fourcc={fourcc_code or 'default'}) ==")

    configs = [
        (None, "no-exposure-set"),
        ((0.25, -4), "ae=0.25,ev=-4"),
        ((0.25, -6), "ae=0.25,ev=-6"),
        ((0.25, -7), "ae=0.25,ev=-7"),
        ((0.25, -8), "ae=0.25,ev=-8"),
        ((0.25, -9), "ae=0.25,ev=-9"),
        ((0.25, -10), "ae=0.25,ev=-10"),
        ((0.25, -11), "ae=0.25,ev=-11"),
        ((0.25, -13), "ae=0.25,ev=-13"),
        ((1, -7), "ae=1,ev=-7"),
        ((3, -7), "ae=3,ev=-7"),
    ]

    results = []
    for exp, desc in configs:
        print(f"  Testing exposure({desc})...", end=" ", flush=True)
        r = benchmark_capture(device_id, backend, backend_name,
                              f"P4: {desc}", duration_s=2.5,
                              fourcc_code=fourcc_code, set_exposure=exp)
        if r:
            print(f"{r.fps:.1f} fps")
            results.append(r)
        else:
            print("FAILED")
        time.sleep(0.3)

    for r in results:
        print(f"  {r.summary_line()}")
    return results


def phase5_convert_rgb(device_id: int, backend: int, backend_name: str,
                       fourcc_code: Optional[str] = None,
                       exposure: Optional[Tuple[float, float]] = None) -> List[TimingResult]:
    """Phase 5: Test with CONVERT_RGB disabled."""
    print(f"\n== Phase 5: CONVERT_RGB toggle ==")

    results = []
    for convert, desc in [(True, "RGB-ON"), (False, "RGB-OFF"), (None, "RGB-default")]:
        print(f"  Testing {desc}...", end=" ", flush=True)
        r = benchmark_capture(device_id, backend, backend_name,
                              f"P5: {desc}", duration_s=3.0,
                              fourcc_code=fourcc_code, set_exposure=exposure,
                              convert_rgb=convert)
        if r:
            print(f"{r.fps:.1f} fps")
            results.append(r)
        else:
            print("FAILED")
        time.sleep(0.3)

    for r in results:
        print(f"  {r.summary_line()}")
    return results


def phase6_buffer_sizes(device_id: int, backend: int, backend_name: str,
                        fourcc_code: Optional[str] = None,
                        exposure: Optional[Tuple[float, float]] = None,
                        convert_rgb: Optional[bool] = None) -> List[TimingResult]:
    """Phase 6: Buffer size variations."""
    print(f"\n== Phase 6: Buffer size variations ==")

    results = []
    for bsz in [1, 2, 3, 4, None]:
        desc = f"buf={bsz}" if bsz is not None else "buf=default"
        print(f"  Testing {desc}...", end=" ", flush=True)
        r = benchmark_capture(device_id, backend, backend_name,
                              f"P6: {desc}", duration_s=3.0,
                              fourcc_code=fourcc_code, set_exposure=exposure,
                              convert_rgb=convert_rgb, buffer_size=bsz)
        if r:
            print(f"{r.fps:.1f} fps")
            results.append(r)
        else:
            print("FAILED")
        time.sleep(0.3)

    for r in results:
        print(f"  {r.summary_line()}")
    return results


def phase7_sustained(device_id: int, backend: int, backend_name: str,
                     duration_s: float = DEFAULT_SUSTAINED_DURATION_S,
                     fourcc_code: Optional[str] = None,
                     exposure: Optional[Tuple[float, float]] = None,
                     convert_rgb: Optional[bool] = None,
                     buffer_size: Optional[int] = 1) -> Optional[TimingResult]:
    """
    Phase 7: Sustained run with per-frame jitter analysis.
    This is the definitive test - captures for `duration_s` and reports
    detailed timing statistics.
    """
    print(f"\n== Phase 7: Sustained {duration_s:.0f}s run ==")
    print(f"  Config: backend={backend_name}, fourcc={fourcc_code or 'default'}, "
          f"convert_rgb={convert_rgb}, buffer={buffer_size}")

    r = benchmark_capture(
        device_id, backend, backend_name,
        f"P7: Sustained {duration_s:.0f}s",
        duration_s=duration_s,
        warmup_frames=120,  # full second warmup for sustained test
        fourcc_code=fourcc_code,
        set_exposure=exposure,
        convert_rgb=convert_rgb,
        buffer_size=buffer_size,
    )

    if r is None:
        print("  FAILED: could not open camera")
        return None

    print(f"\n  +------------------------------------------------+")
    print(f"  |  SUSTAINED RUN RESULTS                         |")
    print(f"  +------------------------------------------------+")
    print(f"  |  Duration:     {r.elapsed_s:8.2f} s                    |")
    print(f"  |  Total frames: {r.total_frames:8d}                      |")
    print(f"  |  Average FPS:  {r.fps:8.1f}                      |")
    print(f"  |  Target FPS:   {TARGET_FPS:8d}                      |")
    print(f"  |  Resolution:   {r.resolution[0]}x{r.resolution[1]}                  |")
    print(f"  |  FOURCC:       {r.fourcc:8s}                    |")
    print(f"  +------------------------------------------------+")
    print(f"  |  Frame interval (dt):                          |")
    print(f"  |    Mean:    {r.dt_mean_ms:8.3f} ms  (ideal: {1000/TARGET_FPS:.3f})  |")
    print(f"  |    Median:  {r.dt_median_ms:8.3f} ms                    |")
    print(f"  |    Std:     {r.dt_std_ms:8.3f} ms                    |")
    print(f"  |    Min:     {r.dt_min_ms:8.3f} ms                    |")
    print(f"  |    Max:     {r.dt_max_ms:8.3f} ms                    |")
    print(f"  |    P1:      {r.dt_p1_ms:8.3f} ms                    |")
    print(f"  |    P99:     {r.dt_p99_ms:8.3f} ms                    |")
    print(f"  |  Jitter (s): {r.jitter_ms:7.3f} ms                    |")
    print(f"  |  Est. drops: {r.dropped_estimate:7d}                        |")
    print(f"  +------------------------------------------------+")

    if r.fps >= 118:
        verdict = "PASS - Stable 120 fps achieved"
    elif r.fps >= 110:
        verdict = "MARGINAL - Close but not stable 120"
    elif r.fps >= 90:
        verdict = "WARN - Significantly below 120 fps"
    else:
        verdict = "FAIL - Nowhere near 120 fps"
    print(f"  |  Verdict: {verdict:37s}|")
    print(f"  +------------------------------------------------+")

    return r


# -- Auto-optimize: find best config ------------------------------------
def auto_find_best_config(device_id: int) -> Dict:
    """
    Run through phases 2-6 to find the combination that yields highest FPS.
    Returns a dict with the best settings.
    """
    print("\n" + "=" * 70)
    print("  AUTO-OPTIMIZING: Finding best configuration for 120 fps")
    print("=" * 70)

    best_fps = 0.0
    best_config: Dict = {
        "backend": cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY,
        "backend_name": "DSHOW" if sys.platform == "win32" else "ANY",
        "fourcc": None,
        "exposure": None,
        "convert_rgb": None,
        "buffer_size": 1,
    }

    # Phase 2: Backend
    p2 = phase2_backend_comparison(device_id)
    if p2:
        best_be = max(p2, key=lambda r: r.fps)
        be_map = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY,
                  "V4L2": cv2.CAP_V4L2}
        best_config["backend"] = be_map.get(best_be.backend, cv2.CAP_ANY)
        best_config["backend_name"] = best_be.backend
        best_fps = best_be.fps
        print(f"\n  >> Best backend: {best_be.backend} ({best_be.fps:.1f} fps)")

    be = best_config["backend"]
    bn = best_config["backend_name"]

    # Phase 3: FOURCC
    p3 = phase3_fourcc_formats(device_id, be, bn)
    if p3:
        best_fc = max(p3, key=lambda r: r.fps)
        if best_fc.fps > best_fps + 2:
            raw_fourcc = best_fc.label.split("=")[-1] if "=" in best_fc.label else None
            if raw_fourcc and raw_fourcc != "<default>":
                best_config["fourcc"] = raw_fourcc
            best_fps = best_fc.fps
        print(f"\n  >> Best FOURCC: {best_config['fourcc'] or 'default'} ({best_fps:.1f} fps)")

    # Phase 4: Exposure
    p4 = phase4_exposure_sweep(device_id, be, bn, best_config["fourcc"])
    if p4:
        best_exp = max(p4, key=lambda r: r.fps)
        if best_exp.fps > best_fps:
            desc = best_exp.label.split(": ")[-1] if ": " in best_exp.label else ""
            if "ae=" in desc:
                parts = desc.split(",")
                ae_val = float(parts[0].split("=")[1])
                ev_val = float(parts[1].split("=")[1])
                best_config["exposure"] = (ae_val, ev_val)
            best_fps = best_exp.fps
        print(f"\n  >> Best exposure: {best_config['exposure'] or 'none'} ({best_fps:.1f} fps)")

    # Phase 5: CONVERT_RGB
    p5 = phase5_convert_rgb(device_id, be, bn, best_config["fourcc"], best_config["exposure"])
    if p5:
        best_cvt = max(p5, key=lambda r: r.fps)
        if "RGB-OFF" in best_cvt.label and best_cvt.fps >= best_fps:
            best_config["convert_rgb"] = False
            best_fps = best_cvt.fps
        elif "RGB-ON" in best_cvt.label and best_cvt.fps > best_fps + 2:
            best_config["convert_rgb"] = True
            best_fps = best_cvt.fps
        print(f"\n  >> Best convert_rgb: {best_config['convert_rgb']} ({best_fps:.1f} fps)")

    # Phase 6: Buffer size
    p6 = phase6_buffer_sizes(device_id, be, bn, best_config["fourcc"],
                             best_config["exposure"], best_config["convert_rgb"])
    if p6:
        best_buf = max(p6, key=lambda r: r.fps)
        buf_label = best_buf.label.split("=")[-1] if "=" in best_buf.label else "1"
        try:
            best_config["buffer_size"] = int(buf_label) if buf_label != "default" else None
        except ValueError:
            pass
        if best_buf.fps > best_fps:
            best_fps = best_buf.fps
        print(f"\n  >> Best buffer: {best_config['buffer_size']} ({best_fps:.1f} fps)")

    return best_config


# -- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Arducam OV9281 120 FPS Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="Camera device index (default: auto-detect)")
    parser.add_argument("--scan", action="store_true",
                        help="Scan all devices and exit")
    parser.add_argument("--phase", "-p", type=int, default=None,
                        help="Run only a specific phase (1-7)")
    parser.add_argument("--duration", type=float, default=DEFAULT_SUSTAINED_DURATION_S,
                        help="Duration for sustained test (Phase 7)")
    parser.add_argument("--backend", choices=["dshow", "msmf", "any", "v4l2"],
                        help="Force specific backend for phase 7")
    parser.add_argument("--fourcc", type=str, default=None,
                        help="Force specific FOURCC for phase 7 (e.g. Y800)")
    parser.add_argument("--no-auto", action="store_true",
                        help="Skip auto-optimization, go straight to phase 7 with defaults")
    parser.add_argument("--high-priority", action="store_true",
                        help="Set process to high priority (reduces OS jitter)")

    args = parser.parse_args()

    print("=" * 70)
    print("  Arducam OV9281 - 120 FPS Benchmark Tool")
    print("  Target: 1280x800 @ 120fps")
    print(f"  Platform: {sys.platform}")
    print(f"  OpenCV version: {cv2.__version__}")
    print("=" * 70)

    if args.high_priority:
        set_high_process_priority()

    # Scan mode
    if args.scan:
        print("\nScanning all camera devices...")
        devices = scan_devices()
        if not devices:
            print("  No cameras found!")
        return

    # Auto-detect device if not specified
    device_id = args.device
    if device_id is None:
        print("\nAuto-detecting Arducam device...")
        devices = scan_devices()
        candidates = [d for d in devices if d["tag"] == "ARDUCAM-candidate"]
        if candidates:
            device_id = candidates[0]["idx"]
            print(f"  Auto-selected device {device_id}")
        elif devices:
            device_id = devices[0]["idx"]
            print(f"  No Arducam candidate found, using device {device_id}")
        else:
            print("  No cameras found! Exiting.")
            return
    else:
        print(f"\nUsing device {device_id} (user-specified)")

    all_results: List[TimingResult] = []

    # Single phase mode
    if args.phase is not None:
        be = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        bn = "DSHOW" if sys.platform == "win32" else "ANY"
        if args.backend:
            be_map = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                      "any": cv2.CAP_ANY, "v4l2": cv2.CAP_V4L2}
            be = be_map[args.backend]
            bn = args.backend.upper()

        if args.phase == 1:
            all_results.extend(phase1_raw_baseline(device_id))
        elif args.phase == 2:
            all_results.extend(phase2_backend_comparison(device_id))
        elif args.phase == 3:
            all_results.extend(phase3_fourcc_formats(device_id, be, bn))
        elif args.phase == 4:
            all_results.extend(phase4_exposure_sweep(device_id, be, bn, args.fourcc))
        elif args.phase == 5:
            all_results.extend(phase5_convert_rgb(device_id, be, bn, args.fourcc))
        elif args.phase == 6:
            all_results.extend(phase6_buffer_sizes(device_id, be, bn, args.fourcc))
        elif args.phase == 7:
            r = phase7_sustained(device_id, be, bn, args.duration,
                                 fourcc_code=args.fourcc)
            if r:
                all_results.append(r)
        else:
            print(f"Unknown phase: {args.phase}")

    elif args.no_auto:
        # Skip optimization, run sustained with defaults/overrides
        be = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        bn = "DSHOW" if sys.platform == "win32" else "ANY"
        if args.backend:
            be_map = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                      "any": cv2.CAP_ANY, "v4l2": cv2.CAP_V4L2}
            be = be_map[args.backend]
            bn = args.backend.upper()
        r = phase7_sustained(device_id, be, bn, args.duration,
                             fourcc_code=args.fourcc)
        if r:
            all_results.append(r)

    else:
        # Full auto-optimize + sustained test
        best = auto_find_best_config(device_id)

        print("\n" + "=" * 70)
        print("  BEST CONFIGURATION FOUND:")
        print(f"    Backend:     {best['backend_name']}")
        print(f"    FOURCC:      {best['fourcc'] or 'default'}")
        print(f"    Exposure:    {best['exposure'] or 'default'}")
        print(f"    CONVERT_RGB: {best['convert_rgb']}")
        print(f"    Buffer size: {best['buffer_size']}")
        print("=" * 70)

        # Now run the definitive sustained test with best config
        r = phase7_sustained(
            device_id,
            best["backend"], best["backend_name"],
            duration_s=args.duration,
            fourcc_code=best["fourcc"],
            exposure=best["exposure"],
            convert_rgb=best["convert_rgb"],
            buffer_size=best["buffer_size"],
        )
        if r:
            all_results.append(r)

    # Final summary
    if all_results:
        print("\n" + "=" * 70)
        print("  ALL RESULTS SUMMARY")
        print("=" * 70)
        for r in all_results:
            print(f"  {r.summary_line()}")

        best_overall = max(all_results, key=lambda r: r.fps)
        print(f"\n  Best overall: {best_overall.fps:.1f} fps ({best_overall.label})")

        if best_overall.fps >= 118:
            print("\n  *** 120 FPS target ACHIEVED ***")
        elif best_overall.fps >= 100:
            print(f"\n  Close to target. Gap is {120 - best_overall.fps:.1f} fps.")
            print("  Consider: USB 3.0 port, shorter cable, fewer USB devices on same hub.")
        else:
            print(f"\n  Significant gap: {120 - best_overall.fps:.1f} fps below target.")
            print("  Likely causes:")
            print("    - Auto-exposure stuck at long exposure time (check Windows Camera app)")
            print("    - USB 2.0 port or hub bandwidth saturation")
            print("    - Driver issue (try updating Arducam UVC driver)")
            print("    - Background processes consuming CPU/USB bandwidth")


if __name__ == "__main__":
    main()
