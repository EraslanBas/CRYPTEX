#!/usr/bin/env python3
"""Fit ONE CRYPTEX model on a saved split (scripts/build_split.py).

A thin front end to ``scripts/sweep_grid.py`` for a single configuration and
seed: the same loading, defaults and saved outputs (model, configuration,
history, validation scores). Any other ``sweep_grid.py`` flag can be passed
through, e.g. ``--lasso-lambda 0.05``.

    python scripts/fit_model.py --split-dir <split> --out-dir <dir> -d 20 --seed 0

The test partition is not read; score the chosen model once with
``scripts/read_test.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_grid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--n-factors", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=700.0)
    ap.add_argument("-K", "--n-subsamples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], "--d", str(args.n_factors), "--alpha", str(args.alpha),
                "-K", str(args.n_subsamples), "--seeds", str(args.seed), *rest]
    return sweep_grid.main()


if __name__ == "__main__":
    raise SystemExit(main())
