from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.benchmark_ws_payload import BenchmarkRuntime


def main(iterations: int = 10000) -> None:
    runtime = BenchmarkRuntime()
    started = time.perf_counter()
    for _ in range(iterations):
        runtime.get_state_message()
    elapsed_ms = (time.perf_counter() - started) * 1000
    print("Backend runtime benchmark")
    print(f"iterations: {iterations}")
    print(f"total_ms: {elapsed_ms:.3f}")
    print(f"avg_state_build_ms: {elapsed_ms / iterations:.6f}")


if __name__ == "__main__":
    main()
