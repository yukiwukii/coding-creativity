#!/usr/bin/env python3
import json
from pathlib import Path

# --- hardcoded source and destination ---
IN_PATH = Path("datasets/CodeForce/llama8_summary.json")
OUT_PATH = Path("datasets/CodeForce/llama8_summary.json")
TYPE = 'cove'

def main():
    # 1. Load existing JSON
    if not IN_PATH.exists():
        raise FileNotFoundError(f"{IN_PATH} not found.")
    with open(IN_PATH, "r") as f:
        data = json.load(f)

    # 2. Hard-coded arrays
    BInsHal = [
       [0.0,53.03,70.05,85.492,96,96.907],
        [0.0,54.04,73.604,86.01,95.429,94.845],
        [0.0,52.02,72.081,84.974,95.429,96.907]
    ]
    BFacHal = [
        [86.869,88.889,89.848,91.71,93.143,91.753],
        [86.869,88.889,89.848,91.71,93.143,91.753],
        [86.869,88.889,89.848,91.71,93.143,91.753]
    ]
    BConv = [
        [13.131,3.535,1.015,0.518,0.571,0.0],
        [13.131,3.535,1.523,0.518,1.143,0.0],
        [13.131,3.03,1.523,0.518,0.571,0.0]
    ]
    BDiv = [
        [98.99,98.737,98.866,99.398,99.143,97.423],
        [98.165,97.643,98.767,98.964,99.486,98.969],
        [97.399,98.232,99.239,98.238,99.771,98.969]
    ]
    BCresc = [
        [13.131,3.535,1.015,0.518,0.571,0.0],
        [13.131,3.535,1.523,0.518,1.143,0.0],
        [12.879,3.03,1.523,0.518,0.571,0.0]
    ]

    # 3. Append new runs
    prefix = f"Llama-3.1-8B-Instruct_sample=199_dp=5_{TYPE}"
    for i in range(len(BInsHal)):
        run_id = f"{prefix}{i+1}"
        if run_id in data:
            print(f"[skip] '{run_id}' already exists")
            continue
        data[run_id] = {
            "type": TYPE,
            "dp_rounds": len(BInsHal[i]) - 1,
            "instruction_hallucination":  BInsHal[i],
            "factual_hallucination":      BFacHal[i],
            "convergent_creativity":      BConv[i],
            "divergent_creativity":       BDiv[i],
            "total_creativity":           BCresc[i]
        }
        print(f"[added] {run_id}")

    # 4. Save back
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated summary written to {OUT_PATH}")

if __name__ == "__main__":
    main()
