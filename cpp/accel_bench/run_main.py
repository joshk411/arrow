#!/usr/bin/env python3
"""Emit the 6-config main-figure CSVs for the Parquet examples.

Every figure in the study uses the same taxonomy:

  1. sync-cpu        stock synchronous CPU path
  2. async-cpu       same restructuring the async path needs, work still on CPU
  3. sync-accel      blocking offload, naive admission (offload ~everything)
  4. async-accel     non-blocking offload, naive admission
  5. opt-sync-accel  blocking offload + tuned admission threshold
  6. opt-async-accel non-blocking offload + tuned admission threshold

Produces pq_main_iaa.csv (gzip compression offloaded to IAA). The page-size and
1-128 core sweeps of the same six systems are in run_scale6.py.
"""

import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "pq_accel_bench")
OUT = os.path.join(HERE, "results")

ITERS = ["--iters", "5"]

# Extra cores handed to the async-cpu control. IAA uses zero extra cores, so
# this number has to be reported alongside the bar.
CPU_ASYNC_THREADS = 8

# Tuned admission thresholds, from the sweeps in this directory:
#   IAA: every page Parquet produces (>=2 KB) is faster on IAA than on zlib,
#        so the tuned threshold is 0 -- admission tuning is a measured no-op.
IAA_TUNED_MIN = 0
IAA_NAIVE_MIN = 0

IAA_BASE = ["--codec", "gzip", "--level", "1"]

IAA_CONFIGS = [
    ("sync-cpu", []),
    ("async-cpu", ["--cpu-async", str(CPU_ASYNC_THREADS)]),
    ("sync-accel", ["--iaa", "sync", "--iaa-min", str(IAA_NAIVE_MIN)]),
    ("async-accel", ["--iaa", "async", "--iaa-min", str(IAA_NAIVE_MIN)]),
    ("opt-sync-accel", ["--iaa", "sync", "--iaa-min", str(IAA_TUNED_MIN)]),
    ("opt-async-accel", ["--iaa", "async", "--iaa-min", str(IAA_TUNED_MIN)]),
]


def run(base, extra):
    cmd = ["sudo", BENCH] + ITERS + base + extra
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.strip().splitlines() if l.strip()]
    header = lines[-2].split(",")
    values = lines[-1].split(",")
    return dict(zip(header, values))


def emit(name, base, configs):
    rows = []
    for config, extra in configs:
        r = run(base, extra)
        if r["verified"] != "1":
            sys.exit(f"{name}/{config}: round-trip verification FAILED")
        r["config"] = config
        r["cmd"] = " ".join(extra) or "(stock)"
        rows.append(r)
        print(f"{name:4s} {config:16s} {float(r['gbps']):7.4f} GB/s  {r['cmd']}")
    path = os.path.join(OUT, f"pq_main_{name}.csv")
    cols = ["config", "cmd"] + [c for c in rows[0] if c not in ("config", "cmd")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    emit("iaa", IAA_BASE, IAA_CONFIGS)
