#!/usr/bin/env python3
"""Page-size sweep and 1-128 writer scaling for the six main-figure systems.

The configurations are imported from run_main.py (IAA_CONFIGS), so every point
is the exact system shown in the main figure (gzip level 1, 4M rows x 4
columns = 128 MB per writer, round-trip verified).

Experiments
  page     1 writer, page size 4 KB .. 4 MB (set exactly with --rows-per-page:
           int64/double PLAIN pages are rows x 8 B). async-cpu is the
           main-figure control (1 writer + 8 pool cores). The tuned admission
           threshold is 0, so opt-* == * and every page goes to IAA.
  threads  1..128 writers, spread round-robin over the six NUMA nodes
           (1,4,0,3,2,5; one per physical core). Each writer pins, then builds
           its own table and output buffer, so memory is node-local; QPL
           sends its jobs to the IAA devices of its own socket (4 per socket).
           async-cpu gets the same core budget as the other systems: T cores
           = W writers + (T-W) zlib pool workers. W in {T/2, T/4, T/8, T/16}
           are all measured (pq_threads6_cpu_candidates.csv) and the best is
           reported. At T=1 there is no pool core, so async-cpu == sync-cpu.

Needs sudo (IAA WQs); the bench is invoked under sudo.
"""
import argparse
import csv
import io
import math
import os
import statistics
import subprocess
import sys

from run_main import BENCH, CPU_ASYNC_THREADS, IAA_BASE, IAA_CONFIGS, OUT

NODE_ORDER = [1, 4, 0, 3, 2, 5]
CORES_PER_NODE = 32
PAGE_KB = [4, 16, 64, 256, 1024, 4096]
THREADS = [1, 2, 4, 8, 16, 32, 64, 128]


def spread_cpus(n):
    """Slot t -> physical core t//6 of node NODE_ORDER[t % 6]."""
    if n > CORES_PER_NODE * len(NODE_ORDER):
        raise SystemExit(f"{n} cores exceeds physical cores")
    return [CORES_PER_NODE * NODE_ORDER[t % 6] + t // 6 for t in range(n)]


def run(args, reps):
    cmd = ["sudo", BENCH] + IAA_BASE + args
    rows = []
    for _ in range(reps):
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print("FAILED:", " ".join(cmd), out.stderr[-800:], file=sys.stderr)
            return None
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        rows.append(next(csv.DictReader(io.StringIO("\n".join(lines[-2:])))))
    med = statistics.median(float(r["gbps"]) for r in rows)
    r = min(rows, key=lambda r: abs(float(r["gbps"]) - med))
    r["cmd"] = " ".join(args)
    return r


def write(path, rows):
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def page_sweep(reps, iters):
    rows = []
    for kb in PAGE_KB:
        rpp = kb * 1024 // 8
        point = ["--iters", str(iters), "--rows-per-page", str(rpp),
                 "--page", str(max(kb * 1024 * 2, 1 << 20)), "--cpu", "40"]
        for name, extra in IAA_CONFIGS:
            r = run(point + extra, reps)
            if r is None:
                continue
            r.update(config=name, page_kb=kb,
                     routed_to="iaa" if "--iaa" in extra else "cpu")
            rows.append(r)
            print(f"page {kb:5d}K {name:16s} {float(r['gbps']):8.4f} GB/s "
                  f"fb={r['iaa_fallbacks']} ver={r['verified']}", flush=True)
    write(os.path.join(OUT, "pq_page6.csv"), rows)


def cpu_candidates(t):
    if t == 1:
        return [(1, 0)]
    out = []
    for div in (2, 4, 8, 16):
        w = max(1, t // div)
        if t - w >= 1 and (w, t - w) not in out:
            out.append((w, t - w))
    return out


def thread_sweep(reps, iters, threads):
    rows, cands = [], []
    for t in threads:
        point = ["--iters", str(iters)]
        cpus = spread_cpus(t)
        for name, extra in IAA_CONFIGS:
            if name == "async-cpu":
                best = None
                for w, p in cpu_candidates(t):
                    if p == 0:
                        args = point + ["--threads", "1", "--cpu-list", str(cpus[0])]
                    else:
                        depth = max(2, 2 * math.ceil(p / w))
                        args = point + ["--threads", str(w),
                                        "--cpu-list", ",".join(map(str, cpus[:w])),
                                        "--cpu-async", str(p),
                                        "--pool-cpus", ",".join(map(str, cpus[w:])),
                                        "--cpu-depth", str(depth)]
                    r = run(args, reps)
                    if r is None:
                        continue
                    r.update(config=name, cores=t, writers=w, pool_workers=p)
                    cands.append(r)
                    print(f"thr {t:4d} async-cpu W={w:<3d} P={p:<3d} "
                          f"{float(r['gbps']):8.3f} GB/s ver={r['verified']}", flush=True)
                    if best is None or float(r["gbps"]) > float(best["gbps"]):
                        best = r
                if best:
                    rows.append(dict(best))
                continue
            args = point + extra + ["--threads", str(t),
                                    "--cpu-list", ",".join(map(str, cpus))]
            r = run(args, reps)
            if r is None:
                continue
            r.update(config=name, cores=t, writers=t, pool_workers=0)
            rows.append(r)
            print(f"thr {t:4d} {name:16s} {float(r['gbps']):8.3f} GB/s "
                  f"fb={r['iaa_fallbacks']} retry={r['iaa_retries']} ver={r['verified']}",
                  flush=True)
    write(os.path.join(OUT, "pq_threads6.csv"), rows)
    write(os.path.join(OUT, "pq_threads6_cpu_candidates.csv"), cands)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["page", "threads", "all"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--threads", default=",".join(map(str, THREADS)))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what in ("page", "all"):
        page_sweep(a.reps, 5)
    if a.what in ("threads", "all"):
        thread_sweep(a.reps, 3, [int(x) for x in a.threads.split(",")])
