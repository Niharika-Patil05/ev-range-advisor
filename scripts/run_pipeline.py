"""Train and evaluate everything.  Usage:  python scripts/run_pipeline.py [--quick]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import train_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller synthetic dataset, faster")
    args = ap.parse_args()
    if args.quick:
        train_all(n_routes=30, trips_per_route=10)
    else:
        train_all()
