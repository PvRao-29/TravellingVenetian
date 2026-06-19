#!/usr/bin/env python3
"""Quick benchmark of OR-Tools modes across seeds (dev helper)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from venice import VeniceTSPConfig, VeniceTSPEnv
from ORtools import solve_with_ortools


def main() -> int:
    base_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    for mode in ("none", "soft", "strict"):
        ok = fail = 0
        for s in range(base_seed, base_seed + num):
            env = VeniceTSPEnv(
                config=VeniceTSPConfig(num_delivery_nodes=15, num_docks=6, seed=s)
            )
            env.reset(seed=s)
            result = solve_with_ortools(
                env, time_window_mode=mode, time_limit_seconds=limit, verbose=False
            )
            if result.success:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {mode} seed={s}: {result.message[:80]}")
        print(f"{mode}: {ok}/{ok + fail} success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
