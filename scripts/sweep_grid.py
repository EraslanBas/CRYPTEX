#!/usr/bin/env python3
"""Train CRYPTEX over a grid of (d, alpha, K, seed) on a saved split.

With no model flags this trains the default configuration (see
``cryptex.train.TrainConfig`` and docs/notes/CURRENT_MODEL.md). Loads
``<split>/trainval`` ONCE and reuses it for every cell. Resumable: a cell whose
row is already in ``sweep_metrics.csv`` (or, with ``--queue``, in ``cells/``) is
skipped.

Per cell it saves everything needed to re-score the model later: the selected
and last epoch's parameters, the configuration, the label encoding, the full
per-epoch history, ``B`` at every evaluation, and the validation logits.

TEST DISCIPLINE. The test partition is never read here; it is read once, at
the end, by ``scripts/read_test.py``.
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cryptex import paths
from cryptex.data.splits import (_pair_key, load_partition,
                                       split_val_per_perturbation, subsample_mask)
from cryptex.objectives import StratifiedNegativeSampler
from cryptex.train import TrainConfig, evaluate, fit, prepare

METRICS = ("auc", "disc", "recon", "accuracy", "lasso_objective", "programs_per_pair",
           "span_residual_median", "real_score_mean", "fake_score_mean")


class _Queue:
    """A work queue that lives in the sweep directory itself.

    claims/<tag>.claim   created with O_CREAT|O_EXCL, so exactly one worker
                         gets a cell; holds host, pid and Slurm step
    cells/<tag>.csv      the finished cell's metrics row (written atomically);
                         a cell is DONE iff this file exists
    sweep_metrics.csv    rebuilt from cells/ under .metrics.lock after each
                         cell, so readers always see a consistent table

    A worker that dies leaves its claim behind. ``clear_stale`` (run at each
    worker's startup, serialised by the same lock) removes claims whose
    process is gone on this host or whose Slurm step no longer exists.
    """

    def __init__(self, root: Path):
        import os, socket
        self.root = root
        self.claims = root / "claims"; self.claims.mkdir(exist_ok=True)
        self.cells = root / "cells"; self.cells.mkdir(exist_ok=True)
        self.lock = root / ".metrics.lock"
        job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
        self.me = {"host": socket.gethostname(), "pid": os.getpid(),
                   "step": f"{job}.{step}" if job and step else None}

    def _locked(self):
        import contextlib, fcntl

        @contextlib.contextmanager
        def cm():
            with open(self.lock, "w") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                yield
        return cm()

    def done(self) -> set[str]:
        return {p.stem for p in self.cells.glob("*.csv")}

    def claim(self, tag: str) -> bool:
        import os
        try:
            fd = os.open(self.claims / f"{tag}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, json.dumps(self.me).encode()); os.close(fd)
        return True

    def finish(self, tag: str, row: dict) -> None:
        tmp = self.cells / f".{tag}.tmp"
        pd.DataFrame([row]).to_csv(tmp, index=False)
        tmp.rename(self.cells / f"{tag}.csv")
        with self._locked():
            rows = [pd.read_csv(p) for p in sorted(self.cells.glob("*.csv"))]
            out = self.root / "sweep_metrics.csv"
            tmpm = self.root / ".sweep_metrics.tmp"
            pd.concat(rows, ignore_index=True).to_csv(tmpm, index=False)
            tmpm.rename(out)
        (self.claims / f"{tag}.claim").unlink(missing_ok=True)

    def _alive(self, c: dict) -> bool:
        import os, subprocess
        if c.get("host") == self.me["host"]:
            try:
                os.kill(int(c["pid"]), 0)
                return True
            except (ProcessLookupError, PermissionError, ValueError):
                return False
        if c.get("step"):
            # Check the JOB, not the step: inside a batch script SLURM_STEP_ID
            # is 0 while squeue lists the step as "<job>.batch", so a step
            # check declared live claims dead and other workers re-trained
            # them. A claim stays until its job ends; a worker that crashes
            # inside a live job keeps its claim until then.
            job = c["step"].split(".")[0]
            # "%A" is the job's own id, also for array tasks (whose "%i" is
            # "<array job>_<task>" and never equals SLURM_JOB_ID, so array
            # workers' claims were declared dead and re-trained)
            r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%A"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                # an ended job is "Invalid job id"; anything else: cannot tell
                return "invalid job id" not in (r.stderr + r.stdout).lower()
            return job in r.stdout.split()
        return True                # other host, no step: cannot tell

    def clear_stale(self) -> None:
        with self._locked():
            done = self.done()
            for p in self.claims.glob("*.claim"):
                try:
                    c = json.loads(p.read_text())
                except (OSError, ValueError):
                    c = {}
                if p.stem in done or not self._alive(c):
                    print(f"queue: clearing stale claim {p.stem} ({c})", flush=True)
                    p.unlink(missing_ok=True)


def _sampler(fa, rows, row_weighted=True):
    return StratifiedNegativeSampler(
        perturbation_idx=fa["perturbation_idx"][rows], context_idx=fa["context_idx"][rows],
        stratum=np.zeros(len(rows), dtype=np.int64),
        n_perturbations=int(fa["perturbation_idx"].max()) + 1,
        n_contexts=int(fa["context_idx"].max()) + 1,
        weights={"same_s_other_pert": 0.5, "same_s_other_context": 0.5},
        row_weighted=row_weighted)


def _val_scores(model, fa, rows, batch=4096, seed=12345) -> dict:
    """Discriminator logits on the validation rows: each row under its own label
    (``pos``) and under a swapped label (``perm``, drawn as in training)."""
    device = next(model.parameters()).device
    neg_p, neg_c = _sampler(fa, rows, model.config.negative_draw == "rows").sample(
        np.arange(len(rows)), np.random.default_rng(seed))
    model.eval()
    t = lambda a: torch.from_numpy(np.asarray(a)).to(device)
    from cryptex.models.group_lasso import group_batches, lasso_groups
    gkey = lasso_groups(fa["perturbation_idx"], fa["context_idx"], model.config.lasso_groups)
    lasso = model.config.coding == "group_lasso"

    def logits(idx, p, c):
        # batches of whole (perturbation, context) pairs for the group lasso;
        # the activities use the row's own pair, the score the label p, c
        out = np.zeros(len(idx), np.float32)
        parts = (group_batches(gkey[idx], batch) if lasso
                 else [np.arange(i, min(i + batch, len(idx))) for i in range(0, len(idx), batch)])
        with torch.no_grad():
            for b in parts:
                j = idx[b]
                cols = t(fa["target_col"][j]) if fa.get("target_col") is not None else None
                z = model.project(t(fa["X"][j]), cols, t(gkey[j]) if lasso else None)
                out[b] = model.score(z, t(p[b]), t(c[b])).cpu().numpy()
        return out

    return {"pos": logits(rows, fa["perturbation_idx"][rows], fa["context_idx"][rows]),
            "perm": logits(rows, neg_p, neg_c)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", type=Path,
                    default=paths.root()/"splits"/"pair_seed0_ctxshuffle_pgenes_disjoint_min100_noMT")
    ap.add_argument("--out-dir", type=Path, default=paths.root()/"sweep_grid")
    ap.add_argument("--d", nargs="+", type=int, default=[20])
    ap.add_argument("--alpha", nargs="+", type=float, default=[700])
    ap.add_argument("-K","--n-subsamples", nargs="+", type=int, default=[10],
                    help="pseudobulk rows per pair used in training (besides the full row)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--min-epochs", type=int, default=None,
                    help="accepted for older job scripts; every model trains --epochs epochs")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--perturbation-half", type=int, choices=[0, 1], default=None,
                    help="train and validate on one random half of the perturbations only "
                         "(all their pairs); adds _half<h>s<seed> to the tag")
    ap.add_argument("--half-seed", type=int, default=0, help="seed of the perturbation halving")
    ap.add_argument("--shared-subspace", action="store_true",
                    help="with --perturbation-half: compute V (PCA) from the training pairs of BOTH "
                         "halves, so models of the two halves share one subspace; adds _sharedV")
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--draw-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--queue", action="store_true",
                    help="shared work queue: any number of workers (GPU or CPU, any host) may "
                         "point at the SAME --out-dir; each cell is claimed atomically "
                         "(claims/<tag>.claim) and its metrics row written to cells/<tag>.csv")
    ap.add_argument("--subspace", nargs="+", default=["fixed"], choices=["free", "fixed"],
                    help="fixed: B = A V; free: B learned directly. A grid dimension")
    ap.add_argument("--basis", nargs="+", default=None, metavar="NAME=PATH",
                    help="external subspaces for the fixed mode, e.g. ica=B_ica.npy (a [d, G] "
                         ".npy); a grid dimension. Default: V from --subspace-method")
    ap.add_argument("--head-basis", nargs="+", default=["linear,square,abs,tanh"],
                    help="statistics the head applies to each factor, one comma-separated set "
                         "per grid value. Non-default sets add _head-<names> to the tag")
    ap.add_argument("--axes-init", default="jd", choices=["random", "jd"],
                    help="start A at the joint-diagonalisation axes (adds _Ajd) or at a random "
                         "rotation per seed (adds _Arandom)")
    ap.add_argument("--label-weight-decay", type=float, default=0.01,
                    help="AdamW weight decay on the label network only; adds _lwd<x>")
    ap.add_argument("--label-dropout", type=float, default=0.2,
                    help="dropout in the label network; adds _ldo<x>")
    ap.add_argument("--embedding-dim", type=int, default=64, help="perturbation embedding size")
    ap.add_argument("--context-embedding-dim", type=int, default=8, help="context embedding size")
    ap.add_argument("--context-embedding-init", choices=["random", "control-means"], default="control-means",
                    help="start context embeddings at random or at the PCA of each context's "
                         "control-cell mean expression (--context-means); adds _ctxctrl")
    ap.add_argument("--context-means", type=Path, default=None,
                    help="contexts x genes control means (default $CRYPTEX_ROOT/context_control_means.parquet, "
                         "from scripts/context_control_means.py)")
    ap.add_argument("--pert-embedding-init", choices=["random", "response"], default="response",
                    help="start perturbation embeddings at random or from how each target "
                         "gene responds across the training pairs; response adds _embresp")
    ap.add_argument("--coding", choices=["ls", "group_lasso"], default="group_lasso",
                    help="activities: the group lasso with one selection of programs per "
                         "(perturbation, context) pair (adds _gl<lambda>), or plain least squares")
    ap.add_argument("--lasso-lambda", type=float, default=0.0272)
    ap.add_argument("--lasso-steps", type=int, default=30)
    ap.add_argument("--lasso-groups", choices=["pair", "row"], default="row",
                    help="rows sharing a program selection: each row alone (default, a lasso per row) "
                         "or the pair (group lasso; runs before 2026-10-28)")
    ap.add_argument("--gene-sparsity", type=float, default=0.05,
                    help="gamma, weight of the L1/L2 sparseness of each program's loadings; adds _gs<gamma>")
    ap.add_argument("--distinct-weight", type=float, default=100.0,
                    help="mu: penalise programs with |cos| above --distinct-tau (adds _dt<tau>)")
    ap.add_argument("--distinct-tau", type=float, default=0.5)
    ap.add_argument("--negative-draw", choices=["rows", "labels"], default="rows",
                    help="swapped labels in proportion to their rows (default), or uniformly "
                         "over the labels (runs before 2026-10-25)")
    ap.add_argument("--subspace-method", choices=["rca", "snr", "pca"], default="pca",
                    help="how V is computed: PCA of the pair means (default), reliable components "
                         "analysis, or signal over the within-pair noise model; adds _pca / _rca<rank> / _snr")
    ap.add_argument("--free-init", choices=["random", "basis"], default="random",
                    help="free mode: B starts at random, or where the fixed model starts "
                         "(A0 V); basis adds _initV<method> to the tag")
    ap.add_argument("--rca-rank", type=int, default=100,
                    help="RCA: R_W truncated to its top RANK eigen-directions")
    ap.add_argument("--noise-rank", type=int, default=50,
                    help="rank of the correlated-noise part of the noise model (snr)")
    ap.add_argument("--noise-max-rows", type=int, default=150_000,
                    help="within-pair residuals used to fit the noise model (snr)")
    ap.add_argument("--no-test", action="store_true",
                    help="accepted for older job scripts; the test partition is never read here")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv = args.out_dir / "sweep_metrics.csv"

    # ONE writer per directory (or the shared --queue): two sweeps writing the
    # same directory would rerun the same cells and overwrite each other's files.
    import fcntl, os
    if args.queue:
        queue = _Queue(args.out_dir)
        queue.clear_stale()
    else:
        queue = None
        lock_path = args.out_dir / ".sweep.lock"
        lock_fh = open(lock_path, "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"{args.out_dir} is locked by another sweep ({lock_path}); "
                "stop it first -- with `scancel <jobid>.<step>`, since killing "
                "the local srun client does not stop an --overlap step")
        lock_fh.write(f"{os.getpid()} {os.uname().nodename}\n"); lock_fh.flush()
    done = set()
    if queue is not None:
        done = queue.done()
        print(f"queue: {len(done)} cells already done in {args.out_dir}", flush=True)
    elif csv.exists():
        prev = pd.read_csv(csv)
        done = set(prev.tag)
        print(f"resuming: {len(done)} cells already done", flush=True)

    # ---------------- load once --------------------------------------
    t0 = time.time()
    Xtv, mtv, genes = load_partition(args.split_dir, "trainval", None)
    shared_bases = None
    if args.perturbation_half is not None and args.shared_subspace:
        # V from the training pairs of both halves; each half's validation pairs are
        # drawn exactly as that half's own run draws them, so none of them enter V
        from cryptex.data.splits import perturbation_half
        from cryptex.data.noise import pca_subspace
        if args.subspace_method != "pca":
            raise SystemExit("--shared-subspace supports --subspace-method pca only")
        pair_all = (mtv.perturbation.astype(str) + "|" + mtv.context.astype(str)).to_numpy()
        shared_train = np.zeros(len(mtv), bool)
        for h_ in (0, 1):
            mh = perturbation_half(mtv.perturbation.to_numpy(), h_, args.half_seed)
            idx = np.nonzero(mh)[0]; m_h = mtv[mh].reset_index(drop=True)
            vp = set(_pair_key(m_h)[split_val_per_perturbation(m_h, args.val_frac, args.val_seed)])
            shared_train[idx[~_pair_key(m_h).isin(vp).to_numpy()]] = True
        shared_bases = {d_: pca_subspace(Xtv, pair_all, shared_train, d_, seed=args.draw_seed)[0]
                        for d_ in sorted(set(args.d))}
        for d_, Vb in shared_bases.items():
            np.save(args.out_dir / f"subspace_V_pca_d{d_}_shared_halves_s{args.half_seed}.npy", Vb)
        print(f"shared subspace from {int(shared_train.sum()):,} training rows of both halves", flush=True)
    if args.perturbation_half is not None:
        from cryptex.data.splits import perturbation_half
        keep_h = perturbation_half(mtv.perturbation.to_numpy(), args.perturbation_half, args.half_seed)
        Xtv, mtv = Xtv[keep_h], mtv[keep_h].reset_index(drop=True)
        print(f"perturbation half {args.perturbation_half} (seed {args.half_seed}): "
              f"{mtv.perturbation.nunique():,} perturbations", flush=True)
    val_real = split_val_per_perturbation(mtv, args.val_frac, args.val_seed)
    val_pairs = set(_pair_key(mtv)[val_real])
    val_all = _pair_key(mtv).isin(val_pairs).to_numpy()
    print(f"trainval {len(mtv):,} rows x {len(genes):,} genes, "
          f"{len(val_pairs):,} val pairs  ({time.time()-t0:.0f}s)", flush=True)

    # ---- the subspace V (train pairs only) ------------------------------------
    bases = {}
    if set(args.subspace) - {"free"} or args.free_init == "basis":
        from cryptex.data.noise import (label_subspace, noise_covariance, pair_means,
                                        pca_subspace, rca_subspace, within_pair_residuals)
        t1 = time.time()
        train_real = ~val_all
        pair = (mtv.perturbation.astype(str) + "|" + mtv.context.astype(str)).to_numpy()
        if args.subspace_method == "snr":
            R = within_pair_residuals(Xtv, pair, train_real, max_rows=args.noise_max_rows,
                                      seed=args.draw_seed)
            noise_model = noise_covariance(R, rank=args.noise_rank, seed=args.draw_seed)
            del R
            np.savez(args.out_dir / "noise_model.npz", diag=noise_model.diag,
                     U=noise_model.U, s=noise_model.s)
            _, means, counts = pair_means(Xtv, pair, train_real)
        for d_ in sorted(set(args.d)):
            if args.subspace_method == "pca":
                bases[d_], ev = pca_subspace(Xtv, pair, train_real, d_, seed=args.draw_seed)
                np.save(args.out_dir / f"subspace_V_pca_d{d_}.npy", bases[d_])
            elif args.subspace_method == "rca":
                bases[d_], ev = rca_subspace(Xtv, pair, train_real, d_, rank=args.rca_rank,
                                             seed=args.draw_seed)
                np.save(args.out_dir / f"subspace_V_rca{args.rca_rank}_d{d_}.npy", bases[d_])
                np.save(args.out_dir / f"rca_reliability_rca{args.rca_rank}_d{d_}.npy", ev)
            else:
                bases[d_], ev = label_subspace(means, counts, noise_model, d_)
                np.save(args.out_dir / f"subspace_V_snr_d{d_}.npy", bases[d_])
        print(f"subspaces ({args.subspace_method}) for d={sorted(bases)} from the "
              f"train pairs  ({time.time()-t1:.0f}s)", flush=True)
        if shared_bases is not None:
            bases = shared_bases
            print("using the subspace shared by both halves", flush=True)

    # NAME=PATH loads an external subspace
    ext_bases = {}
    for kv in (args.basis or []):
        name, _, path = kv.partition("=")
        ext_bases[name] = np.load(path).astype(np.float32) if path else None
    basis_names = list(ext_bases) or [None]
    cells = [c for c in itertools.product(args.d, args.alpha, args.n_subsamples, args.seeds,
                                          args.subspace, basis_names, args.head_basis)
             if c[4] == "fixed" or c[5] == basis_names[0]]
    if queue is not None:
        # longest first (cost grows with K)
        cells.sort(key=lambda c: -c[2])
    print(f"{len(cells)} cells, {len(cells)-len(done)} to run\n", flush=True)

    for i, (d, a, K, seed, sub, bname, hbasis) in enumerate(cells, 1):
        head = tuple(hbasis.split(","))
        named = bname is not None and sub == "fixed"
        use_ext = named and ext_bases[bname] is not None
        tag = (f"d{d}_a{a:g}_K{K}_seed{seed}" + ("" if sub == "free" else f"_{sub}")
               + (f"_{bname}" if named else "")
               + ("" if head == ("linear", "square", "abs", "tanh") else f"_head-{'-'.join(head)}")
               + (f"_A{args.axes_init}" if sub == "fixed" else "")
               + ((f"_initV{args.subspace_method}" + ("_Arandom" if args.axes_init == "random" else "")
                   if args.free_init == "basis" else "") if sub == "free"
                  else {"snr": "_snr", "pca": "_pca"}.get(args.subspace_method, f"_rca{args.rca_rank}"))
               + (f"_lwd{args.label_weight_decay:g}" if args.label_weight_decay else "")
               + (f"_ldo{args.label_dropout:g}" if args.label_dropout else "")
               + ("_embresp" if args.pert_embedding_init == "response" else "")
               + ("_ctxctrl" if args.context_embedding_init == "control-means" else "")
               + f"_emb{args.embedding_dim}x{args.context_embedding_dim}"
               + (f"_gs{args.gene_sparsity:g}" if args.gene_sparsity else "")
               + ((f"_dt{args.distinct_tau:g}" + (f"_dw{args.distinct_weight:g}" if args.distinct_weight != 100 else ""))
                  if args.distinct_weight else "")
               + ((f"_gl{args.lasso_lambda:g}" if args.lasso_groups == "pair" else f"_rowl{args.lasso_lambda:g}")
                  if args.coding == "group_lasso" else "_ls")
               + ("_negrows" if args.negative_draw == "rows" else "")
               + (f"_half{args.perturbation_half}s{args.half_seed}"
                  + ("_sharedV" if args.shared_subspace else "")
                  if args.perturbation_half is not None else ""))
        # re-read each time: a cell finished by any concurrent process is never repeated
        if queue is not None:
            if tag in queue.done() or not queue.claim(tag):
                continue
        else:
            if csv.exists():
                done = set(pd.read_csv(csv, usecols=["tag"]).tag)
            if tag in done:
                continue
        t = time.time()
        keep = subsample_mask(mtv, K, seed=args.draw_seed) | val_all
        X, meta, val = Xtv[keep], mtv[keep].reset_index(drop=True), val_all[keep]
        fa = prepare(X, meta, genes)
        plev, clev = fa.pop("perturbation_levels"), fa.pop("context_levels")
        ctx_init = None
        if args.context_embedding_init == "control-means":
            from cryptex.train import control_mean_embeddings
            cm = pd.read_parquet(args.context_means or paths.root() / "context_control_means.parquet")
            ctx_init = control_mean_embeddings(cm, clev, args.context_embedding_dim)
        cfg = TrainConfig(n_factors=d, alpha=a, seed=seed,
                          epochs=args.epochs, batch_size=args.batch_size,
                          device=args.device, log_every=0,
                          eval_every=args.eval_every, subspace=sub, basis=head,
                          axes_init=args.axes_init,
                          subspace_method=args.subspace_method, rca_rank=args.rca_rank,
                          noise_rank=args.noise_rank, noise_max_rows=args.noise_max_rows,
                          free_init=args.free_init, pert_embedding_init=args.pert_embedding_init,
                          embedding_dim=args.embedding_dim,
                          context_embedding_dim=args.context_embedding_dim,
                          label_weight_decay=args.label_weight_decay,
                          label_dropout=args.label_dropout,
                          gene_sparsity=args.gene_sparsity, distinct_weight=args.distinct_weight,
                          distinct_tau=args.distinct_tau,
                          coding=args.coding, lasso_lambda=args.lasso_lambda,
                          lasso_steps=args.lasso_steps, lasso_groups=args.lasso_groups,
                          negative_draw=args.negative_draw)
        snaps: dict[int, np.ndarray] = {}
        model, hist = fit(config=cfg, train_rows=~val, val_rows=val,
                          subspace_basis=(ext_bases[bname] if use_ext else bases.get(d)),
                          context_embedding_init=ctx_init,
                          on_eval=lambda ep, m: snaps.__setitem__(ep, m.loading_matrix().copy()),
                          **fa)
        pd.DataFrame(hist).to_csv(args.out_dir/f"history_{tag}.csv", index=False)
        np.save(args.out_dir/f"B_{tag}.npy", model.loading_matrix())
        ep = np.array(sorted(snaps))
        np.savez(args.out_dir/f"B_evals_{tag}.npz", epoch=ep,
                 B=np.stack([snaps[e] for e in ep]).astype(np.float32))
        # everything needed to re-score the model later: the selected and the
        # last epoch's parameters, and the configuration they were trained with
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   args.out_dir/f"model_best_{tag}.pt")
        torch.save(model.final_state_, args.out_dir/f"model_final_{tag}.pt")
        import dataclasses
        (args.out_dir/f"config_{tag}.json").write_text(json.dumps(
            {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()},
             "selected_epoch": int(model.selected_epoch_), "final_epoch": int(model.final_epoch_),
             "n_perturbations": int(fa["perturbation_idx"].max()) + 1,
             "n_contexts": int(fa["context_idx"].max()) + 1, "n_genes": int(fa["X"].shape[1]),
             "split_dir": str(args.split_dir), "tag": tag}, indent=1, default=str))
        np.save(args.out_dir/f"levels_{tag}.npy",
                np.array([plev, clev], dtype=object), allow_pickle=True)
        np.savez(args.out_dir/f"val_scores_{tag}.npz", **_val_scores(model, fa, np.nonzero(val)[0]))

        X_t = torch.from_numpy(fa["X"]); col_t = torch.from_numpy(fa["target_col"])
        row = {"tag": tag, "d": d, "alpha": a, "K": K, "seed": seed,
               "subspace": sub, "head_basis": ",".join(head), "basis": (bname if named else ("V" if sub != "free" else "")),
               "subspace_method": args.subspace_method, "rca_rank": args.rca_rank,
               "free_init": args.free_init, "pert_embedding_init": args.pert_embedding_init,
               "embedding_dim": args.embedding_dim, "context_embedding_dim": args.context_embedding_dim,
               "context_embedding_init": args.context_embedding_init,
               "label_weight_decay": args.label_weight_decay, "label_dropout": args.label_dropout,
               "axes_init": args.axes_init,
               "gene_sparsity": args.gene_sparsity, "distinct_weight": args.distinct_weight,
               "distinct_tau": args.distinct_tau,
               "coding": args.coding, "lasso_lambda": args.lasso_lambda,
               "lasso_steps": args.lasso_steps, "lasso_groups": args.lasso_groups,
               "negative_draw": args.negative_draw,
               "perturbation_half": args.perturbation_half, "half_seed": args.half_seed,
               "shared_subspace": args.shared_subspace,
               "epochs_run": len(hist), "seconds": round(time.time()-t, 1)}
        for name, rows_ in (("train", np.nonzero(~val)[0]), ("val", np.nonzero(val)[0])):
            m = evaluate(model, X_t, fa["perturbation_idx"], fa["context_idx"], rows_,
                         _sampler(fa, rows_, args.negative_draw == "rows"),
                         np.random.default_rng(12345), col_t, None)
            for k in METRICS:
                row[f"{name}_{k}"] = m.get(k)
        if queue is not None:
            queue.finish(tag, row)
        else:
            # append safely: re-read + concat keeps one consistent header
            allr = pd.concat([pd.read_csv(csv), pd.DataFrame([row])], ignore_index=True) \
                   if csv.exists() else pd.DataFrame([row])
            allr.to_csv(csv, index=False)
        print(f"[{i}/{len(cells)}] {tag:<26} "
              f"disc {row['train_disc']:.4f}/{row['val_disc']:.4f}  "
              f"recon {row['train_recon']:.6f}/{row['val_recon']:.6f}  "
              f"val auc {row['val_auc']:.4f}  {row['seconds']:.0f}s", flush=True)
    (args.out_dir/"sweep_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print("\nSWEEP DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
