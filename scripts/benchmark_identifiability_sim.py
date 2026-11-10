#!/usr/bin/env python3
"""Ground-truth identifiability benchmark: CRYPTEX (fixed and free) vs PCA, ICA.

Adapted from a collaborator's EXTRACT_identifiability_sim.py: the simulation,
validation split and metrics are theirs; this version runs the current
CRYPTEX code in both modes and adds a realistic-noise option.

    PYTHONPATH=src python scripts/benchmark_identifiability_sim.py --out sim_bench

Generative model (defaults in brackets)
---------------------------------------
B_true   D x G programs [6 x 400], each loading 100 random genes, N(0,1) weights,
         unit-norm rows; programs overlap.
a_p      perturbation engagement P x D [200 x 6]: nonzero with probability
         --active-frac [0.5], Laplace (unit variance) or Gaussian.
s_pc     C contexts [6]. additive: a_p + 0.5 g_c + 0.3 d_pc;
         multiplicative: a_p * m_c + 0.3 d_pc, m_ck ~ LogNormal(0, 0.7).
         ammi: a_p + 0.5 g_c + --interaction [1] * f_p * h_c + 0.3 d_pc, main
         effects plus a per-program interaction; f_p has a_p's sparsity
         pattern, h_ck ~ N(0, 1).
rows     x = amp * s_pc @ B_true + noise; cells n ~ LogNormal(log 300, 0.7) in
         [80, 3000]; full-data row noise has covariance Sigma_cell / n; K [4]
         subsample rows of sizes m in [50, n) add independent noise with
         covariance Sigma_cell (1/m - 1/n) -- the exact law of a subset mean.
noise    isotropic: Sigma_cell = 1.5^2 I (the collaborator's setting).
         realistic: per-gene SDs 1.5 * LogNormal(0, --gene-sd-spread [1.0])
         plus --shared-noise [3] directions shared across genes, each with
         variance --shared-var [20] times the mean per-gene variance.

Methods (all scored against B_true)
-----------------------------------
CRYPTEX fixed   the default model: V from the training rows, B = A V.
CRYPTEX free    B learned directly; alpha = c / mean(x^2), c in --alphas.
CRYPTEX <lm>    fixed mode with a structured label model, lm in --label-models:
                product lambda_k = f_k(p) g_k(c); ammi a_k(p) + b_k(c) + f_k(p) g_k(c).
V (frozen)      the label-driven subspace with its initial axes, no training.
PCA, FastICA    scikit-learn on training full-data rows (ICA: 3 seeds).
null            mean MCC of 50 random rotations of the PCA basis.

Extra regimes: --families gaussian --active-frac 1.0 with the multiplicative
structure gives Gaussian sources whose per-program gains depend on the context
(a CP tensor structure: identifiable from the labels, hard for ICA); with the
additive structure nothing is identifiable (control). --program-corr w > 0
makes programs co-engaged, breaking ICA's independence assumption.

Metrics: MCC (Hungarian-matched |correlation| of loadings with B_true),
subspace overlap with span(B_true), cross-seed MCC, validation accuracy.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GENES_PER_PROGRAM, SIGMA_CELL = 100, 1.5


# ---------------------------------------------------------------- simulation

def make_B(rng, D, G):
    B = np.zeros((D, G))
    for k in range(D):
        idx = rng.choice(G, GENES_PER_PROGRAM, replace=False)
        B[k, idx] = rng.normal(0, 1, GENES_PER_PROGRAM)
    return B / np.linalg.norm(B, axis=1, keepdims=True)


def draw(rng, family, size):
    if family == "gaussian":
        return rng.normal(0, 1, size)
    if family == "laplace":
        return rng.laplace(0, 1 / np.sqrt(2), size)
    raise ValueError(family)


def make_noise(rng, G, a):
    """A sampler for cell-level noise: returns f(scale) -> one G-vector with
    covariance scale^2 * Sigma_cell."""
    if a.noise == "isotropic":
        return lambda s: rng.normal(0, SIGMA_CELL * s, G)
    gene_sd = SIGMA_CELL * rng.lognormal(0, a.gene_sd_spread, G)
    U, _ = np.linalg.qr(rng.normal(size=(G, a.shared_noise)))
    shared_sd = np.sqrt(a.shared_var * np.mean(gene_sd ** 2))
    def f(s):
        return s * (gene_sd * rng.normal(size=G) + U @ (shared_sd * rng.normal(size=a.shared_noise)))
    return f


def simulate(family, structure, seed, a):
    G, D, P, C, K = a.genes, a.programs, a.perts, a.contexts, a.subsamples
    rng = np.random.default_rng(seed)
    B = make_B(rng, D, G)
    w = a.program_corr
    if w > 0:
        # correlated programs: a per-perturbation component shared by all its
        # programs, both in which programs are engaged and in how strongly
        u = (1 - w) * rng.random((P, D)) + w * rng.random((P, 1))
        val = (1 - w) * draw(rng, family, (P, D)) + w * draw(rng, family, (P, 1))
        val = val / np.sqrt((1 - w) ** 2 + w ** 2)          # back to unit variance
        act = val * (u < a.active_frac)
    else:
        act = draw(rng, family, (P, D)) * (rng.random((P, D)) < a.active_frac)
    if structure == "additive":
        S = act[:, None, :] + 0.5 * draw(rng, family, (C, D))[None, :, :]
    elif structure == "multiplicative":
        S = act[:, None, :] * np.exp(rng.normal(0, 0.7, (C, D)))[None, :, :]
    elif structure == "ammi":
        f = draw(rng, family, (P, D)) * (act != 0)
        S = (act[:, None, :] + 0.5 * draw(rng, family, (C, D))[None, :, :]
             + a.interaction * f[:, None, :] * rng.normal(size=(C, D))[None, :, :])
    else:
        raise ValueError(structure)
    S = S + 0.3 * draw(rng, family, (P, C, D))
    mean = a.amp * np.einsum("pcd,dg->pcg", S, B)
    noise = make_noise(rng, G, a)

    rows, pi, ci, st, nc = [], [], [], [], []
    for p in range(P):
        for c in range(C):
            n = int(np.clip(rng.lognormal(np.log(300), 0.7), 80, 3000))
            e_full = noise(1 / np.sqrt(n))
            rows.append(mean[p, c] + e_full)
            pi.append(p); ci.append(c); st.append(0); nc.append(n)
            for k, m in enumerate(np.sort(rng.integers(50, n, K))):
                rows.append(mean[p, c] + e_full + noise(np.sqrt(max(1 / m - 1 / n, 0))))
                pi.append(p); ci.append(c); st.append(k + 1); nc.append(m)
    return B, np.array(rows, np.float32), *(np.array(v) for v in (pi, ci, st, nc))


def split(pi, ci, P, C, rng, frac=0.12):
    """Pair-level validation mask; each perturbation keeps >= 3 training pairs."""
    pairs = sorted(set(zip(pi.tolist(), ci.tolist())))
    rng.shuffle(pairs)
    left = {p: C for p in range(P)}
    val = set()
    for p, c in pairs:
        if len(val) >= frac * len(pairs):
            break
        if left[p] > 2:
            val.add((p, c)); left[p] -= 1
    v = np.array([(p, c) in val for p, c in zip(pi, ci)])
    return ~v, v


# ---------------------------------------------------------------- metrics

def subspace_overlap(Bt, Be):
    Qt, _ = np.linalg.qr(Bt.T)
    Qe, _ = np.linalg.qr(Be.T)
    return float(np.mean(np.linalg.svd(Qt.T @ Qe, compute_uv=False) ** 2))


def mcc(Bt, Be):
    return float(match_factors(Bt, Be).mcc)


def cross_seed(Bs):
    return float(np.mean([mcc(x, y) for x, y in itertools.combinations(Bs, 2)])) if len(Bs) > 1 else None


# ---------------------------------------------------------------- fitting

def run_fit(a, X, pi, ci, st, nc, tr, va, seed, **kw):
    # simulated perturbations have no target genes: random embedding starts
    cfg = TrainConfig(n_factors=a.programs, epochs=a.epochs, eval_every=a.eval_every,
                      seed=seed, log_every=0, noise_rank=a.noise_rank,
                      pert_embedding_init="random", **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        model, hist = fit(X, pi, ci, config=cfg, train_rows=tr, val_rows=va)
    best = max((h for h in hist if "val_auc" in h), key=lambda h: h["val_auc"])
    return model, best


def summarise(Bt, fits):
    Bs = [m.loading_matrix() for m, _ in fits]
    return {"mcc": [round(mcc(Bt, b), 3) for b in Bs],
            "overlap": [round(subspace_overlap(Bt, b), 3) for b in Bs],
            "cross_seed": None if len(Bs) < 2 else round(cross_seed(Bs), 3),
            "val_auc": [round(h["val_auc"], 3) for _, h in fits],
            "selected_epoch": [int(h["epoch"]) for _, h in fits]}


def run_condition(family, structure, a):
    t0 = time.time()
    B, X, pi, ci, st, nc = simulate(family, structure, a.data_seed, a)
    tr, va = split(pi, ci, a.perts, a.contexts, np.random.default_rng(1))
    full_tr = X[tr & (st == 0)]
    pca = PCA(a.programs).fit(full_tr)
    icas = [FastICA(a.programs, whiten="unit-variance", max_iter=2000, random_state=s).fit(full_tr).mixing_.T
            for s in a.seeds]
    rng = np.random.default_rng(2)
    null = float(np.mean([mcc(B, np.linalg.qr(rng.normal(size=(a.programs,) * 2))[0] @ pca.components_)
                          for _ in range(50)]))
    res = {"family": family, "structure": structure, "noise": a.noise, "amp": a.amp,
           "active_frac": a.active_frac, "program_corr": a.program_corr,
           "pca": {"mcc": round(mcc(B, pca.components_), 3),
                   "overlap": round(subspace_overlap(B, pca.components_), 3)},
           "fastica": {"mcc": [round(mcc(B, b), 3) for b in icas],
                       "overlap": round(subspace_overlap(B, icas[0]), 3),
                       "cross_seed": round(cross_seed(icas), 3)},
           "random_rotation_null": round(null, 3)}

    fixed = [run_fit(a, X, pi, ci, st, nc, tr, va, s, subspace="fixed") for s in a.seeds]
    res["extract_fixed"] = summarise(B, fixed)
    fixed_r = [run_fit(a, X, pi, ci, st, nc, tr, va, s, subspace="fixed", axes_init="random")
               for s in a.seeds]
    res["extract_fixed_Arandom"] = summarise(B, fixed_r)
    V = fixed[0][0].subspace_basis_
    res["V_frozen_axes"] = {"mcc": round(mcc(B, V), 3), "overlap": round(subspace_overlap(B, V), 3)}
    ms = float(np.mean(X[tr] ** 2))
    for c in (a.alphas if a.free else []):
        res[f"extract_free_c{c:g}"] = summarise(
            B, [run_fit(a, X, pi, ci, st, nc, tr, va, s, subspace="free", alpha=c / ms) for s in a.seeds])
    res["seconds"] = round(time.time() - t0)
    return res


def show(r):
    print(f"\n== {r['family']} / {r['structure']} / noise={r['noise']} amp={r['amp']} "
          f"active={r['active_frac']} corr={r['program_corr']}  ({r['seconds']}s)")
    print(f"  {'method':<22} {'MCC vs truth':<24} {'subspace':<10} {'cross-seed':<10} val acc")
    print(f"  {'PCA':<22} {r['pca']['mcc']:<24} {r['pca']['overlap']:<10}")
    f = r["fastica"]
    print(f"  {'FastICA':<22} {str(f['mcc']):<24} {f['overlap']:<10} {f['cross_seed']:<10}")
    print(f"  {'V (frozen axes)':<22} {r['V_frozen_axes']['mcc']:<24} {r['V_frozen_axes']['overlap']:<10}")
    for k, v in r.items():
        if k.startswith("extract_"):
            print(f"  {k:<22} {str(v['mcc']):<24} {np.mean(v['overlap']):<10.3f} "
                  f"{str(v['cross_seed']):<10} {v['val_auc']}")
    print(f"  random-rotation null  {r['random_rotation_null']}")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sim_bench")
    ap.add_argument("--families", nargs="+", default=["laplace", "gaussian"])
    ap.add_argument("--structures", nargs="+", default=["additive", "multiplicative"],
                    choices=["additive", "multiplicative", "ammi"])
    ap.add_argument("--noise", choices=["isotropic", "realistic"], default="isotropic")
    ap.add_argument("--gene-sd-spread", type=float, default=1.0)
    ap.add_argument("--shared-noise", type=int, default=3)
    ap.add_argument("--shared-var", type=float, default=20.0)
    ap.add_argument("--amp", type=float, default=3.0, help="signal amplitude (collaborator's default 3)")
    ap.add_argument("--active-frac", type=float, default=0.5)
    ap.add_argument("--program-corr", type=float, default=0.0,
                    help="weight of a per-perturbation component shared by all programs "
                         "(0 = independent programs, as in ICA's model)")
    ap.add_argument("--genes", type=int, default=400)
    ap.add_argument("--programs", type=int, default=6)
    ap.add_argument("--perts", type=int, default=200)
    ap.add_argument("--contexts", type=int, default=6)
    ap.add_argument("--subsamples", type=int, default=4)
    ap.add_argument("--noise-rank", type=int, default=10, help="shared-noise rank CRYPTEX fits for V")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 10.0], help="free mode: alpha = c / mean(x^2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--free", action=argparse.BooleanOptionalAction, default=True,
                    help="also fit the free model")
    ap.add_argument("--interaction", type=float, default=1.0,
                    help="ammi structure: scale of the interaction f_p * h_c")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    global TrainConfig, fit, match_factors, PCA, FastICA
    import torch
    from cryptex.train import TrainConfig, fit
    from cryptex.evaluation.stability import match_factors
    from sklearn.decomposition import PCA, FastICA
    torch.set_num_threads(a.threads)
    warnings.filterwarnings("ignore", category=UserWarning)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = []
    for family in a.families:
        for structure in a.structures:
            r = run_condition(family, structure, a)
            show(r); results.append(r)
    tag = f"{a.noise}_amp{a.amp:g}_active{a.active_frac:g}_corr{a.program_corr:g}_{'-'.join(a.families)}"
    if a.structures != ["additive", "multiplicative"]:
        tag += f"_{'-'.join(a.structures)}" + (f"_int{a.interaction:g}" if "ammi" in a.structures else "")
    (out / f"bench_{tag}.json").write_text(json.dumps({"settings": vars(a), "results": results}, indent=1))
    print(f"\nwrote {out / f'bench_{tag}.json'}")


if __name__ == "__main__":
    main()
