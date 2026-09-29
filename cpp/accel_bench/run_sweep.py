#!/usr/bin/env python3
"""Sweeps the Parquet writer accel benchmark (needs sudo for /dev/dsa, /dev/iax).

Writes results/pq_results.csv (one row per configuration; each row is already
the median over --iters write iterations inside the benchmark).
"""
import argparse
import csv
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "pq_accel_bench")

GZIP_BASE = [
    ("zlib-1", dict(codec="gzip", level=1)),
    ("zlib-6", dict(codec="gzip", level=6)),
    ("iaa-sync", dict(codec="gzip", iaa="sync")),
    ("iaa-async-d1", dict(codec="gzip", iaa="async", depth=1)),
    ("iaa-async-d2", dict(codec="gzip", iaa="async", depth=2)),
    ("iaa-async-d4", dict(codec="gzip", iaa="async", depth=4)),
    ("iaa-async-d8", dict(codec="gzip", iaa="async", depth=8)),
    ("iaa-async-d16", dict(codec="gzip", iaa="async", depth=16)),
]
OTHER_CODECS = [
    ("none", dict(codec="none")),
    ("snappy", dict(codec="snappy")),
    ("zstd-1", dict(codec="zstd", level=1)),
    ("zstd-3", dict(codec="zstd", level=3)),
]


def points(exp):
    if exp in ("iaa", "all"):
        for name, p in GZIP_BASE + OTHER_CODECS:
            yield "iaa", name, dict(p)
        for name, p in [("iaa-sync", dict(codec="gzip", iaa="sync")),
                        ("iaa-async-d8", dict(codec="gzip", iaa="async", depth=8))]:
            yield "huffman", name + "-fixed", dict(p, huffman="fixed")
        for d in [1, 2, 4, 8, 16]:
            yield "progress", f"iaa-async-d{d}-front", dict(codec="gzip", iaa="async", depth=d,
                                                           progress="front")
    if exp in ("page", "all"):
        for page in [65536, 262144, 1 << 20]:
            for name, p in [("zlib-1", dict(codec="gzip", level=1)),
                            ("iaa-sync", dict(codec="gzip", iaa="sync")),
                            ("iaa-async-d8", dict(codec="gzip", iaa="async", depth=8))]:
                yield "page", name, dict(p, page=page)
    if exp in ("dict", "all"):
        for name, p in [("zlib-1", dict(codec="gzip", level=1)),
                        ("zstd-1", dict(codec="zstd", level=1)),
                        ("iaa-sync", dict(codec="gzip", iaa="sync")),
                        ("iaa-async-d8", dict(codec="gzip", iaa="async", depth=8))]:
            yield "dict+str", name, dict(p, dictionary=1, **{"str-cols": 1})
    if exp in ("dsa", "all"):
        for batch in [1024, 8192, 65536]:
            for d in ["off", "sync", "async"]:
                yield "dsa", f"dsa-{d}", dict(codec="none", batch=batch, dsa=d, **{"dsa-min": 4096})
        for d in ["off", "sync", "async"]:
            yield "dsa+iaa", f"dsa-{d}+iaa-async", dict(codec="gzip", iaa="async", depth=8,
                                                        dsa=d, **{"dsa-min": 4096})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="all")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for exp, name, p in points(args.experiment):
        cmd = ["sudo", BIN, "--iters", str(args.iters)]
        for k, v in p.items():
            cmd += [f"--{k}", str(v)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print("FAILED:", " ".join(cmd), out.stderr[-2000:], file=sys.stderr)
            continue
        for r in csv.DictReader(io.StringIO(out.stdout)):
            r = {"experiment": exp, "config": name, **r}
            rows.append(r)
            print(exp, name, r["gbps"], "GB/s ratio", r["ratio"], "fallbacks",
                  r["iaa_fallbacks"], "verified", r["verified"], flush=True)
    suffix = "" if args.experiment == "all" else f"_{args.experiment}"
    path = os.path.join(args.out, f"pq_results{suffix}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
