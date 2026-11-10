import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def save(result: dict, name: str, outdir: str = "results") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    acc = result.get("test_accuracy", result.get("accuracy_pooled", result.get("best_score")))
    print(f"\nSaved {path}" + (f" | headline: {acc:.4f}" if isinstance(acc, float) else ""))
    return path


def subjects_arg(text: str):
    """'1-70' or '1-109' or '1,2,5' -> list of ints."""
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data-path", default=None, help="MNE data cache (default ~/mne_data)")
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=42)
    return p
