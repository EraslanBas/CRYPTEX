#!/usr/bin/env python3
"""The single test read -- the only place the test partition is opened.

Configurations are chosen on validation only. Once one is chosen, run this for
it: it loads the saved model (the epoch validation selected), scores every test
row under its own label and under a swapped label, and prints the metrics.
Nothing is refitted.

Every read is appended to ``<sweep>/TEST_READS.log`` with a timestamp. Test is
meant to be read once; the log makes a second read visible rather than silent,
and this script refuses a repeat unless ``--again`` is given. A test number
obtained after choosing among configurations by their test scores is no longer
an unbiased estimate.

    python scripts/read_test.py <sweep_dir> --tag <tag> --reason "chosen on val AUC"
"""
from __future__ import annotations
import argparse, datetime, getpass, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract.data.splits import load_partition
from extract.objectives import StratifiedNegativeSampler
from extract.train import evaluate, load_model, prepare

SHOW = ["auc", "accuracy", "disc", "recon", "lasso_objective", "programs_per_pair",
        "span_residual_median"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--tag", nargs="+", required=True,
                    help="the configuration(s) chosen ON VALIDATION")
    ap.add_argument("--reason", default="",
                    help="why these configs -- recorded in the log")
    ap.add_argument("--again", action="store_true",
                    help="allow a read when test has been read before")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    log = args.sweep_dir / "TEST_READS.log"
    prior = log.read_text().strip().splitlines() if log.exists() else []
    if prior and not args.again:
        print(f"test has already been read {len(prior)} time(s):")
        for line in prior:
            print(f"  {line}")
        raise SystemExit("refusing a repeat read; pass --again if this is deliberate")
    missing = [t for t in args.tag if not (args.sweep_dir / f"config_{t}.json").exists()]
    if missing:
        raise SystemExit(f"no saved model for: {missing}")

    cache = {}
    for tag in args.tag:
        cfg = json.loads((args.sweep_dir / f"config_{tag}.json").read_text())
        split = cfg["split_dir"]
        if split not in cache:
            cache[split] = load_partition(split, "test", None)
        Xte, mte, genes = cache[split]
        plev, clev = np.load(args.sweep_dir / f"levels_{tag}.npy", allow_pickle=True)
        fa = prepare(Xte, mte, genes, perturbation_levels=plev, context_levels=clev)
        model = load_model(args.sweep_dir, tag, "best", device=args.device)
        rows = np.arange(len(mte))
        sampler = StratifiedNegativeSampler(
            perturbation_idx=fa["perturbation_idx"], context_idx=fa["context_idx"],
            stratum=np.zeros(len(rows), dtype=np.int64),
            n_perturbations=len(plev), n_contexts=len(clev),
            weights={"same_s_other_pert": 0.5, "same_s_other_context": 0.5},
            # as the model was trained; configs saved before the option drew uniformly
            row_weighted=cfg.get("negative_draw", "labels") == "rows")
        m = evaluate(model, torch.from_numpy(fa["X"]), fa["perturbation_idx"], fa["context_idx"],
                     rows, sampler, np.random.default_rng(12345),
                     torch.from_numpy(fa["target_col"]), None)
        print(f"{tag}\n  " + "  ".join(f"test_{k} {m[k]:.4f}" for k in SHOW if k in m))

    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(log, "a") as fh:
        fh.write(f"{stamp}\t{getpass.getuser()}\t{','.join(args.tag)}\t{args.reason}\n")
    print(f"\nlogged to {log}  (read #{len(prior) + 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
