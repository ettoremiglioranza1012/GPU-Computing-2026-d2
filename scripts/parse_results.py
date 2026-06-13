#!/usr/bin/env python3
"""parse_results.py — parse strong-scaling SLURM outputs, compute speedup/efficiency.

Usage (from repo root):
    python3 scripts/parse_results.py

Outputs:
    results_tables/strong_scaling_raw.csv      per-rank measurements (all 30 files)
    results_tables/strong_scaling_summary.csv  per (matrix, P): times, speedup, efficiency
"""

import re, os, glob, csv
from collections import defaultdict

OUTPUTS_DIR  = "outputs"
RESULTS_DIR  = "results_tables"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_file(path):
    with open(path) as f:
        text = f.read()

    r = {"path": path, "per_rank": []}

    m = re.search(r"Correctness: (\w+)", text)
    r["correct"] = m.group(1) if m else "UNKNOWN"

    m = re.search(r"Matrix: (\S+)\s+nrows=(\d+)\s+ncols=(\d+)\s+nnz=(\d+)\s+P=(\d+)", text)
    if not m:
        return None
    r["matrix"] = m.group(1)
    r["nrows"]  = int(m.group(2))
    r["nnz"]    = int(m.group(4))
    r["P"]      = int(m.group(5))

    def grab_float(pattern):
        m2 = re.search(pattern, text)
        return float(m2.group(1)) if m2 else None

    r["max_kernel_ms"]    = grab_float(r"Wall-clock kernel time\s+\(max across ranks\):\s+([\d.]+)")
    r["max_comm_ms"]      = grab_float(r"Wall-clock Allgather time \(max across ranks\):\s+([\d.]+)")
    r["total_ms"]         = grab_float(r"Total SpMV time \(comm \+ compute\):\s+([\d.]+)")
    r["comm_pct"]         = grab_float(r"Comm fraction:\s+([\d.]+)%")
    r["agg_gflops"]       = grab_float(r"Aggregate GFLOPs.*?:\s+([\d.]+)")

    m = re.search(
        r"NNZ balance.*?min:\s*(\d+)\s+avg:\s*([\d.]+)\s+max:\s*(\d+)\s+imbalance:\s*([\d.]+)%",
        text)
    if m:
        r["nnz_min"]       = int(m.group(1))
        r["nnz_avg"]       = float(m.group(2))
        r["nnz_max"]       = int(m.group(3))
        r["nnz_imbalance"] = float(m.group(4))

    # First GPU listed
    m = re.search(r"GPUs\s*:\s*(.+)", text)
    r["gpu"] = m.group(1).strip() if m else "unknown"

    # Per-rank CSV rows
    HDR = "rank,matrix,P,local_nnz,local_rows,kernel_ms,comm_ms,comm_vol_MB,mem_MB,bandwidth_GBs,gflops"
    if HDR in text:
        block = text[text.index(HDR) + len(HDR):]
        for line in block.strip().splitlines():
            line = line.strip()
            if not line or not line[0].isdigit():
                break
            parts = line.split(",")
            if len(parts) == 11:
                r["per_rank"].append({
                    "rank":          int(parts[0]),
                    "matrix":        parts[1],
                    "P":             int(parts[2]),
                    "local_nnz":     int(parts[3]),
                    "local_rows":    int(parts[4]),
                    "kernel_ms":     float(parts[5]),
                    "comm_ms":       float(parts[6]),
                    "comm_vol_MB":   float(parts[7]),
                    "mem_MB":        float(parts[8]),
                    "bandwidth_GBs": float(parts[9]),
                    "gflops":        float(parts[10]),
                })
    return r

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "strong_*.out")))
    if not files:
        print(f"No strong_*.out files found in {OUTPUTS_DIR}/")
        return

    parsed = []
    for f in files:
        r = parse_file(f)
        if r:
            parsed.append(r)
        else:
            print(f"  [SKIP] could not parse {os.path.basename(f)}")

    print(f"Parsed {len(parsed)}/{len(files)} output files")

    # ── Raw per-rank CSV ──────────────────────────────────────────────────
    raw_path = os.path.join(RESULTS_DIR, "strong_scaling_raw.csv")
    raw_fields = ["rank","matrix","P","local_nnz","local_rows",
                  "kernel_ms","comm_ms","comm_vol_MB","mem_MB","bandwidth_GBs","gflops"]
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for r in parsed:
            for row in r["per_rank"]:
                w.writerow(row)
    print(f"Written: {raw_path}")

    # ── Summary: one row per (matrix, P) ─────────────────────────────────
    by_matrix = defaultdict(dict)
    for r in parsed:
        by_matrix[r["matrix"]][r["P"]] = r

    # Warn about GPU heterogeneity
    het_warned = set()
    for matrix, runs in by_matrix.items():
        gpus = {P: r["gpu"] for P, r in runs.items()}
        unique = set(gpus.values())
        if len(unique) > 1 and matrix not in het_warned:
            print(f"  [NOTE] {matrix}: GPU types differ across P values — {gpus}")
            het_warned.add(matrix)

    summary_rows = []
    for matrix in sorted(by_matrix):
        runs   = by_matrix[matrix]
        t1     = (runs.get(1) or {}).get("total_ms")
        t1k    = (runs.get(1) or {}).get("max_kernel_ms")   # kernel-only baseline
        gpu_p1 = (runs.get(1) or {}).get("gpu", "?")

        for P in sorted(runs):
            r = runs[P]
            total  = r.get("total_ms")
            kernel = r.get("max_kernel_ms")

            speedup_total  = round(t1  / total,  4) if (t1  and total)  else (1.0 if P == 1 else None)
            speedup_kernel = round(t1k / kernel, 4) if (t1k and kernel) else (1.0 if P == 1 else None)
            eff_total      = round(speedup_total  / P, 4) if speedup_total  else None
            eff_kernel     = round(speedup_kernel / P, 4) if speedup_kernel else None

            summary_rows.append({
                "matrix":          matrix,
                "P":               P,
                "nnz":             r.get("nnz"),
                "nrows":           r.get("nrows"),
                "max_kernel_ms":   r.get("max_kernel_ms"),
                "max_comm_ms":     r.get("max_comm_ms"),
                "total_ms":        total,
                "comm_pct":        r.get("comm_pct"),
                "agg_gflops":      r.get("agg_gflops"),
                "nnz_min":         r.get("nnz_min"),
                "nnz_avg":         r.get("nnz_avg"),
                "nnz_max":         r.get("nnz_max"),
                "nnz_imbalance":   r.get("nnz_imbalance"),
                "speedup_total":   speedup_total,
                "efficiency_total":eff_total,
                "speedup_kernel":  speedup_kernel,
                "efficiency_kernel": eff_kernel,
                "gpu":             r.get("gpu"),
                "gpu_p1":          gpu_p1,
                "correct":         r.get("correct"),
            })

    summary_path = os.path.join(RESULTS_DIR, "strong_scaling_summary.csv")
    summary_fields = [
        "matrix","P","nnz","nrows",
        "max_kernel_ms","max_comm_ms","total_ms","comm_pct","agg_gflops",
        "nnz_min","nnz_avg","nnz_max","nnz_imbalance",
        "speedup_total","efficiency_total","speedup_kernel","efficiency_kernel",
        "gpu","gpu_p1","correct",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Written: {summary_path}")

    # ── Print table ───────────────────────────────────────────────────────
    print()
    print(f"{'Matrix':<20} {'P':>2}  {'NNZ':>10}  {'kern_ms':>8}  {'comm_ms':>8}  "
          f"{'total_ms':>9}  {'comm%':>5}  {'S_tot':>6}  {'E_tot':>6}  {'S_kern':>6}  {'GPU'}")
    print("─" * 115)
    prev = None
    for row in summary_rows:
        if prev and prev != row["matrix"]:
            print()
        prev = row["matrix"]
        flag = " !" if row.get("gpu") != row.get("gpu_p1") and row["P"] > 1 else "  "
        print(
            f"{row['matrix']:<20} {row['P']:>2}  {row['nnz']:>10,}  "
            f"{row['max_kernel_ms']:>8.4f}  {row['max_comm_ms']:>8.4f}  "
            f"{row['total_ms']:>9.4f}  {row['comm_pct']:>4.1f}%  "
            f"{row['speedup_total'] or 0:>6.3f}  {row['efficiency_total'] or 0:>6.3f}  "
            f"{row['speedup_kernel'] or 0:>6.3f}  "
            f"{row['gpu']}{flag}"
        )
    print()
    print("! = GPU type differs from P=1 baseline (L40S vs A30)")
    print("  speedup_total  = T(P=1,total) / T(P,total)   [comm+compute]")
    print("  speedup_kernel = T(P=1,kernel) / T(P,kernel) [compute only, same GPU arch effect]")

if __name__ == "__main__":
    main()
