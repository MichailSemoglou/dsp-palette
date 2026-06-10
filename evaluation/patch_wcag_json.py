"""
patch_wcag_json.py — Patch stored result JSONs with corrected WCAG values.

Uses wcag_recheck.csv (already computed) to update only the rows that changed.
Writes patched JSONs in-place.

Usage
-----
    python3 -m research.evaluation.patch_wcag_json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_RECHECK_CSV   = _ROOT / "results/ablation/wcag_recheck.csv"
_RESULTS_RAW   = _ROOT / "results/raw"
_ABLATION_RAW  = _ROOT / "results/ablation/raw"


def patch_file(json_path: pathlib.Path, method: str, new_aa: float, new_aaa: float) -> None:
    data = json.loads(json_path.read_text())
    entry = data["methods"].get(method)
    if entry is None:
        print(f"  WARN: method {method!r} not found in {json_path.name}")
        return
    metrics = entry.setdefault("metrics", {})
    old_aa  = metrics.get("wcag_aa_coverage")
    old_aaa = metrics.get("wcag_aaa_coverage")
    metrics["wcag_aa_coverage"]  = round(new_aa,  6)
    metrics["wcag_aaa_coverage"] = round(new_aaa, 6)
    json_path.write_text(json.dumps(data, indent=2))
    old_aa_s  = f"{old_aa:.4f}"  if old_aa  is not None else "N/A"
    old_aaa_s = f"{old_aaa:.4f}" if old_aaa is not None else "N/A"
    print(f"  Patched {json_path.name} [{method}]: "
          f"aa {old_aa_s}→{new_aa:.4f}, aaa {old_aaa_s}→{new_aaa:.4f}")


def main() -> None:
    df = pd.read_csv(_RECHECK_CSV)

    # Only process rows where something changed
    changed = df[df["aa_changed"] | df["aaa_changed"]]
    print(f"Patching {len(changed)} (image, method) pairs …\n")

    for _, row in changed.iterrows():
        image_id = row["image_id"]
        method   = row["method"]
        new_aa   = float(row["recomp_aa"])
        new_aaa  = float(row["recomp_aaa"])

        if row["source"] == "raw":
            json_path = _RESULTS_RAW / f"{image_id}.json"
        else:  # ablation
            json_path = _ABLATION_RAW / f"{image_id}.json"

        if not json_path.exists():
            print(f"  WARN: {json_path} not found — skipping")
            continue

        patch_file(json_path, method, new_aa, new_aaa)

    print("\nDone.")


if __name__ == "__main__":
    main()
