#!/usr/bin/env python3
"""Mean log-normalised expression of each context's control cells (the 100k
the logFCs were computed against), over the genes of a split; then PCA of the
contexts x genes matrix. Writes $CRYPTEX_ROOT/context_control_means.parquet,
which sweep_grid.py --context-embedding-init control-means reads.

    python scripts/context_control_means.py"""
import sys, glob, numpy as np, pandas as pd, anndata as ad
from concurrent.futures import ProcessPoolExecutor
C = '/chimera-cold-storage/ctc/large_storage/ctc/beraslan/ModuleFinder/computese/controls/'
R = '/large_storage/ctc/beraslan/ModuleFinder/'
gl = pd.read_csv(R + 'splits/pair_seed0_ctxshuffle_pgenes/gene_list.tsv', sep='\t', comment='#')
genes = gl.gene[gl.keep].astype(str).to_numpy()
def one(path):
    a = ad.read_h5ad(path, backed='r')
    idx = pd.Index(a.var_names.astype(str)).get_indexer(genes)
    assert (idx >= 0).all()
    acc = np.zeros(len(genes)); n = 0
    for s in range(0, a.n_obs, 10_000):
        x = a.X[s:s + 10_000]; x = x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
        acc += x[:, idx].sum(0); n += len(x)
    a.file.close()
    return path.split('/')[-1].replace('_controls_100000.h5ad', ''), acc / n, n
files = sorted(glob.glob(C + '*_controls_100000.h5ad'))
with ProcessPoolExecutor(8) as ex:
    res = list(ex.map(one, files))
ctx = [r[0] for r in res]; M = np.stack([r[1] for r in res]); print('cells per context:', sorted({r[2] for r in res}))
pd.DataFrame(M, index=ctx, columns=genes).to_parquet(R + 'context_control_means.parquet')
def show(A, label):
    s = np.linalg.svd(A, compute_uv=False) ** 2; f = s / s.sum()
    print(f'\n{label}: variance explained per component and cumulative')
    print('  PC  ' + ' '.join(f'{i+1:>5d}' for i in range(len(f))))
    print('  each' + ' '.join(f'{v:5.3f}' for v in f))
    print('  cum ' + ' '.join(f'{v:5.3f}' for v in np.cumsum(f)))
show(M - M.mean(0), 'PCA, centred across the 16 contexts (standard)')
show(M, 'Uncentred (PC1 is the shared average expression profile)')
Z = (M - M.mean(0)); U, S, Vt = np.linalg.svd(Z, full_matrices=False)
print('\nContexts on PC1, PC2 (centred):')
for c, a, b in sorted(zip(ctx, U[:, 0] * S[0], U[:, 1] * S[1]), key=lambda t: t[1]): print(f'  {c:22s} {a:+7.2f} {b:+7.2f}')
