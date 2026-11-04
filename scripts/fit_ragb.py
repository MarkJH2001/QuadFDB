"""Fit RAGB from an externally prepared, FIT-only physical-window archive."""
import argparse
from pathlib import Path
import numpy as np
from quadfdi import FitBatch, RAGB


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-npz", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(20261009, 20261010, 20261011), default=20261009)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    # No CAL/TEST loader, raw-record parser, or server path is inferred here.
    with np.load(args.fit_npz, allow_pickle=False) as data:
        x, health, ids, roles = (data[k].copy() for k in
                                ("windows", "motor_health", "flight_ids", "roles"))
    if roles.shape != (len(x),) or roles.dtype.kind != "U" or not np.all(roles == "fit"):
        raise ValueError("Every supplied window must explicitly declare role='fit'")
    if ids.dtype.kind not in "iU" or not np.all((health == 0) | (health == 1)):
        raise ValueError("Use integer/Unicode flight IDs and binary motor-health labels")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty checkpoint directory; existing results are not overwritten")
    RAGB(args.seed, args.threads).fit(FitBatch(x, health, ids), output=args.output)
    print(f"Saved portable RAGB checkpoint to {args.output}")


if __name__ == "__main__":
    main()
