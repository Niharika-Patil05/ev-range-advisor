"""Train and evaluate everything.  Usage:  python scripts/run_pipeline.py [--quick] [--vehicle leaf|escooter]

NOTE: this still runs on SYNTHETIC data, whose results are circular by construction
(see docs/00_INVESTIGATION_REPORT.md section 3). It exists to exercise the pipeline.
Real-data experiments arrive in Phase 2 under src/experiments/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import E_SCOOTER_PLACEHOLDER, NISSAN_LEAF_2013  # noqa: E402
from src.pipeline import train_all  # noqa: E402

VEHICLES = {"leaf": NISSAN_LEAF_2013, "escooter": E_SCOOTER_PLACEHOLDER}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller synthetic dataset, faster")
    ap.add_argument("--vehicle", choices=sorted(VEHICLES), default="leaf",
                    help="which vehicle's parameters to use (default: leaf, the primary research vehicle)")
    args = ap.parse_args()
    vehicle = VEHICLES[args.vehicle]
    if args.quick:
        train_all(vehicle, n_routes=30, trips_per_route=10)
    else:
        train_all(vehicle)
