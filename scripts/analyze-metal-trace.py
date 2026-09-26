#!/usr/bin/env python3
"""Export and summarize target-process GPU intervals from a completed xctrace run.

Union intervals before calculating duty time: summing overlapping channels is wrong.
This template reports encoder activity, not per-shader execution time or occupancy.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, help="benchmark --trace output directory")
    args = p.parse_args()
    trace = args.directory / "metal.trace"
    toc_path = args.directory / "trace-toc.xml"
    gpu_path = args.directory / "gpu-intervals.xml"
    subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), "--toc", "--output", str(toc_path)], check=True)
    subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), "--xpath",
                    '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]', "--output", str(gpu_path)], check=True)
    toc = ET.parse(toc_path).getroot()
    target = toc.find("./run/info/target/process")
    started = datetime.fromisoformat(toc.findtext("./run/info/summary/start-date")).timestamp()
    gpu = ET.parse(gpu_path).getroot()
    ids = {e.get("id"): e for e in gpu.iter() if e.get("id")}

    def element(e):
        return ids[e.get("ref")] if e.get("ref") else e

    names = [c.findtext("mnemonic") for c in gpu.findall("./node/schema/col")]
    intervals = []
    for row in gpu.iter("row"):
        columns = dict(zip(names, list(row)))
        proc = element(columns["process"])
        if proc.findtext("pid") != target.get("pid") and proc.get("fmt") != f'{target.get("name")} ({target.get("pid")})':
            continue
        if element(columns["state"]).get("fmt") != "Active":
            continue
        start = int(element(columns["start"]).text) / 1e9
        duration = int(element(columns["duration"]).text) / 1e9
        intervals.append((start, start + duration))
    if not intervals:
        raise SystemExit("no target GPU intervals; inspect raw trace before drawing conclusions")
    windows = []
    for run in json.loads((args.directory / "summary.json").read_text())["runs"]:
        left = run["started_unix"] - started
        right = left + run["generation"]
        active = merged([(max(left, s), min(right, e)) for s, e in intervals if s < right and e > left])
        busy = sum(e - s for s, e in active)
        edges = [left] + [v for interval in active for v in interval] + [right]
        gaps = [edges[i + 1] - edges[i] for i in range(0, len(edges) - 1, 2)]
        windows.append({"index": run["index"], "generation_seconds": run["generation"],
                        "gpu_active_seconds": busy, "gpu_duty_fraction": busy / run["generation"],
                        "largest_gap_seconds": max(gaps), "gaps_over_1ms": sum(g > .001 for g in gaps)})
    report = {"target_pid": target.get("pid"), "raw_interval_count": len(intervals), "windows": windows,
              "interpretation": "Target GPU active interval union, not GPU utilization/occupancy or per-kernel breakdown. Trace overhead is included; use uninstrumented benchmarks for speed claims. Gap includes CPU preparation, synchronization and other processes."}
    (args.directory / "gpu-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
