# CRYPTEX

**Identifiable discovery of gene programs from perturbation-by-context screens.**

CRYPTEX finds the gene programs behind a perturbation screen that was run in
several contexts (for example, the same CRISPR perturbations in different cell
types). It learns a small set of programs, groups of genes that move together,
shared by the whole screen, and tells you which programs each perturbation
engages in each context, and how strongly.

## The data

Each data point is one perturbation in one context: the change in expression of
every gene, compared with unperturbed control cells from the same context, as a
log-fold change. Every (perturbation, context) pair contributes several such
rows: one from all its cells, and one from each of several **disjoint groups
of cells of a fixed size** (pseudobulks). Because the groups share no cells, the differences
between a pair's rows show how much its response would change if other cells had
been measured: the measurement noise of that pair.

A perturbation's logFC is an estimate, not an exact value, and with few cells
random cell-to-cell variation alone can produce large apparent changes. Pairs
with too few cells are therefore left out. The model is never told how
many cells a row has or which row is the full one: all rows of a pair are treated
alike, so nothing the model uses would be missing for new data. Genes that no
perturbation changes are removed, and so are the mitochondrial transcripts;
nothing else is rescaled.

## The model

A measured response is assumed to be a mix of a few gene programs:
response ≈ activities × programs, where the programs are the rows of a matrix
$\mathbf B$ (programs × genes) and the activities say how strongly this
perturbation, in this context, engages each program. The number of programs,
$d$, is a hyperparameter chosen for each dataset.

**Where the programs live.** The programs are combinations of the $d$ main
directions of the screen's responses: the principal components $\mathbf V$ of
the pairs' average responses. Averaging a pair's rows removes most of their
measurement noise, so these directions follow what perturbations do. Inside this
space the programs are $\mathbf B = \mathbf A\mathbf V$, and only the small
matrix $\mathbf A$ is learned.

**Where they start.** The contexts change how strongly perturbations vary
along each program. CRYPTEX starts from the axes along which these differences
between contexts are cleanest (joint diagonalisation of the contexts'
covariances). These axes are found from the data alone, before training.

**Few programs per response.** A perturbation is expected to act on a few
processes, not on everything. CRYPTEX therefore describes every response with
as few programs as it can (a lasso), and a perturbation may use different
programs in different contexts. Each row is fitted on its own, so it is one
data point: its activities depend on that row alone, not on the other rows of
its pair. Axes on which responses need few programs are cheaper, which helps
fix the programs.

**Which programs.** The programs are also chosen by a discriminator. It is shown
a response together with a (perturbation, context) label and asked: does this
response belong to this label? The false examples are real responses with
another label. For each program it compares the response's activity with what
the label implies, adds this evidence over all programs, and decides. Because
each program is judged on its own, programs that correspond to separate
processes make the task easier than mixtures of them.

**Few genes per program, and distinct programs.** A gene program is expected
to involve a limited set of genes. CRYPTEX therefore rewards programs whose
weight is concentrated on few genes (up- or down-regulated), and, so that
several programs cannot crowd onto the same genes, it forbids any two programs
from pointing in nearly the same direction. Programs may still share genes.

## Training and choosing a model

Training minimises the discriminator's loss plus the lasso objective
(reconstruction error plus the penalty on the number of programs) and the two
gene-axis terms. The data are
split by (perturbation, context) pair into training, validation and test sets; a
validation or test pair is never seen during training. All models in a
comparison train for the same number of epochs, and each is kept at the epoch where it best ranks true labels above wrong
ones on validation pairs (validation ROC AUC). Models are compared on validation
only, over several random seeds, and the test pairs are used once, at the end.

## Reading the results

- **Programs:** each row of $\mathbf B$ is one program, a weight for every gene
  in log-fold-change units. Its top genes say what the program is.
- **Activities:** for every response (row), which programs it uses and how far
  each moves: up, down, or exactly zero for the programs it does not use. A
  (perturbation, context) pair's activities are the average over its rows.
- **Effects:** activities split into what a perturbation does everywhere, what a
  context does on its own, and the **interaction**: where a perturbation's effect
  on a program depends on the context.

The [Supplement](#supplement) gives the full method: the data processing, the
subspace and starting axes, the lasso, the architecture and objective,
negatives, training, evaluation, configuration and the code layout.

## Supplement

### Model overview

Each row is a vector $\mathbf x \in \mathbb R^G$ of `ashr`-shrunken log-fold
changes for one perturbation $p$ in one context $c$, measured against controls
from the same context. The model assumes the rows are linear mixtures of $d$
latent programs, $\mathbf x \approx \mathbf z\mathbf B$, and wants the axes of
$\mathbf z$ to be the programs rather than an arbitrary rotation of them.

This is two problems: *which subspace* the programs span, and *which axes*
inside it are the programs. Any rotation of the axes inside a subspace fits the
data equally well, so the axes need information beyond the fit. CRYPTEX uses
three sources of it:

1. **The subspace** is the span of the top principal components $\mathbf V$
   ($d \times G$) of the training pairs' mean responses, computed once and fixed.
2. **The starting axes** come from how the contexts differ: the joint
   diagonalisation of the per-context covariances of the pair means inside
   $\mathbf V$.
3. **The axes are then refined** by criteria that all depend on them: a lasso
   that asks each response to use few programs, a discriminator that may only
   score each program separately, and a prior on the programs themselves (few
   genes each, no two nearly collinear).

$$
\mathbf B = \mathbf A\,\mathbf V, \qquad
\mathcal L = \mathcal L_{\text{disc}} + \alpha\,\mathcal L_{\text{lasso}}
+ \gamma\,\mathcal L_{\text{genes}} + \mu\,\mathcal L_{\text{distinct}}
$$

with only $\mathbf A$ ($d \times d$), the label network and the discriminator's
bias learned.

### What a row is

The model never sees counts. The `de/` package turns cell-level single-cell
data into one matrix per context:

1. **Plan the rows.** Each (perturbation, context) pair gets a full-data row from
   all its cells and up to $K$ pseudobulk rows, each from $m$ cells drawn without
   replacement, so that no cell is in two pseudobulks (`--n-subsamples`,
   `--replicate-cells`).
2. **Fix the control set.** One control subset per context (`--n-control-cells`) is
   drawn once, saved, and shared by every row, so the control term of the
   standard error is identical across rows.
3. **Welch statistics.** On the $\log(1+\text{CPM})$ scale, per gene,
   $\hat\Delta_g = \bar y^{\,p}_g - \bar y^{\,\text{ctrl}}_g$ and
   $\widehat{\mathrm{se}}_g = \sqrt{s^{2,p}_g/n_p + s^{2,\text{ctrl}}_g/n_{\text{ctrl}}}$.
4. **Shrink.** `ashr` runs on $(\hat\Delta_g, \widehat{\mathrm{se}}_g)$, with one
   prior shared by all rows of a context; it conditions on the standard error,
   so rows of different precision are handled in one fit. The row $\mathbf x$ is
   the vector of posterior means.
5. **Assemble.** Each context's posterior means become one rows × genes matrix.

Pairs with fewer cells than one pseudobulk needs are removed from the analysis
(`--min-pair-cells`): their responses are dominated by sampling noise. The cell counts are used only here, to build
the matrices. In the model, all rows of a pair are **exchangeable**: it is not
told which row is the full-data one, how many cells a row has, or which
pseudobulk it is. Cell counts would not exist for new data, and their noise
level would let a classifier recognise perturbations without learning their
biology.

#### The gene axis

No centring or per-gene scaling is applied anywhere. `ashr` has already put
every entry on a common logFC scale; dividing by per-gene standard deviation
would promote low-variance genes. Genes that no perturbation moves are dropped:
a gene is kept if $|x| > t$ in at least $n_{\min}$ (perturbation, context)
pairs, counted on the full-data rows of the training pairs (`--threshold`,
`--min-affected`). Every perturbed gene is kept as well, so that each target's own response
is available (it starts the perturbation embeddings, below). The mitochondrial
transcripts (`MT-*`) are excluded: their responses are very large and strongly
correlated with each other, mostly reflecting cell stress and quality rather
than a program, and without scaling they would dominate any sparsity measure on
the gene axis.

#### The on-target mask

Knocking down gene X drops X's own transcript by roughly its control mean.
These entries sit many standard errors from zero and can carry a large share
of a row's squared norm. Left in, they would claim a factor that encodes knockdown
efficiency. Each row is mapped to its own gene's column (`NO_MASK = −1` for
controls or unmatched names), and that column is excluded from the activities,
the reconstruction error and the span residual alike. A compound screen simply
has no masked entry.

### The subspace $\mathbf V$

$\mathbf V$ is the space the programs are allowed to live in. It is computed
once, before training, from the measured rows of the training pairs, and does
not change during training.

1. **Average each pair.** The mean of a pair's rows (its full-data row and its
   pseudobulks, all weighted alike) is its response with most of the
   measurement noise averaged out.
2. **PCA.** $\mathbf V$ holds the top $d$ right singular vectors of the matrix
   of pair means (pairs × genes), as unit-length rows.

The pair means are **not centred**: for log-fold changes, zero already means "no
response", and centring would shift every gene by its average response. Working
on pair means rather than individual rows makes the components follow
differences *between* perturbations, which is what the programs describe; the
within-pair scatter is measurement noise.

$\mathbf V$ depends on which perturbations the screen contains. Computed
separately on two disjoint halves of the perturbations, the two subspaces
overlap only partly, and programs found on different data can agree only as
far as their subspaces do. Comparing models trained on different perturbations
therefore separates two questions: how stable the subspace is, and how stably
the axes are chosen inside a given subspace (both halves using one
$\mathbf V$).

`fit()` computes $\mathbf V$ automatically (`cryptex.data.noise.pca_subspace`); a
precomputed basis can also be passed in.

### Starting axes: joint diagonalisation across contexts

Inside $\mathbf V$, every invertible $\mathbf A$ gives the same subspace. The
starting $\mathbf A$ comes from a second-order property of the data that
depends on the axes: how the variance of the programs across perturbations
differs between contexts.

If the programs are independent sources whose spread across perturbations is
context-specific (a context in which a pathway is silenced shrinks that
program's variation, another amplifies it), then in each context $c$ the covariance of
the pair means, written in $\mathbf V$'s coordinates, is

$$
\boldsymbol\Sigma_c = \mathbf M\,\mathbf D_c\,\mathbf M^\top ,
$$

with the same mixing $\mathbf M$ for every context and a context-specific
diagonal $\mathbf D_c$. One matrix $\mathbf M$ that makes all the contexts' covariances
diagonal at once identifies the programs, as long as the contexts' variance
profiles differ (second-order blind source separation).

1. Write the training pair means in an orthonormal basis $\mathbf Q$ of
   $\mathbf V$'s span.
2. For every context, compute the covariance $\boldsymbol\Sigma_c$ of the pair
   means over its pairs (centred within the context). Contexts with no more
   pairs than $d$ are left out, since their covariance is not of full rank.
3. Whiten by their average, $\mathbf W = \big(\tfrac1C\sum_c\boldsymbol\Sigma_c\big)^{-1/2}$,
   and find the orthogonal $\mathbf U$ that makes every
   $\mathbf W\boldsymbol\Sigma_c\mathbf W$ as diagonal as possible (Jacobi
   joint diagonalisation, Cardoso & Souloumiac 1996).
4. The programs are the columns of the mixing $\mathbf W^{-1}\mathbf U$:
   $\mathbf B_0 = \mathbf U^\top\mathbf W^{-1}\mathbf Q^\top$, rows scaled to
   unit length, and $\mathbf A_0 = \mathbf B_0\mathbf V^{+}$.

This start is deterministic and uses only training data. How much signal is
behind it can be checked by computing it separately on two halves of the
perturbations and comparing the axes. Training then refines the axes
with the two criteria below (`cryptex.train.jd_axes`, `axes_init="jd"`).

### Activities: a lasso per row

The activities of a row are not a plain projection. For each row $\mathbf x$,
CRYPTEX solves

$$
\min_{\mathbf z}\;
\tfrac12\big\|\mathbf m\odot(\mathbf x-\mathbf z\mathbf B)\big\|^2
\;+\;\lambda\sum_{k=1}^{d}|z_k| ,
$$

where $\mathbf m$ masks the row's own target gene.

- **One row, one data point.** Each row is fitted on its own: its activities
  depend on that row and on $\mathbf B$ only, not on the other rows of its
  pair or on how many there are. The discriminator therefore judges every row
  independently, a new single response can be encoded in the same way, and
  the number of rows of a pair (which follows its cell count) cannot reach the
  activities.
- **Exact zeros.** The penalty sets activities exactly to zero, so each row
  uses a few programs and leaves the others at zero. A perturbation's rows
  usually select similar programs because they measure the same response, but
  nothing forces them to.
- **Unit-length rows of $\mathbf B$** stop the penalty from being avoided by
  scaling a program up and its activities down.
- **$\lambda$** sets how many programs a row uses, and is tuned on validation.
  A natural scale for it is the typical size of a program's activity: the
  median over rows and programs of $|z_k|$ for least-squares activities on the
  starting axes. Larger $\lambda$ gives sparser rows, which favours
  reproducible axes but leaves less information for the discriminator.

**Solving it.** FISTA (proximal gradient with momentum), for all rows of a
batch in parallel: start at the least-squares activities, take a gradient step
on the reconstruction error with step $1/L$ ($L$ the largest eigenvalue of
$\mathbf B\mathbf B^\top$), shrink each activity by soft-thresholding,
$z_k \leftarrow \operatorname{sign}(z_k)\max\!\big(0,\,|z_k|-\eta\lambda\big)$,
and repeat for a fixed number of steps (`lasso_steps`). The steps are
differentiable, so gradients reach $\mathbf B$. The same activities are used
everywhere: in the reconstruction objective and by the discriminator, so the
whole model rests on one fit.

**Why it helps fix the axes.** If the programs are the axes, a response lies
on a few of them and needs few non-zero activities. Rotate the axes and the
same response spreads over many programs, which costs more penalty. The lasso
objective therefore prefers axes on which perturbations act sparsely, which is
also what is expected biologically.

The same solver can instead share one selection of programs across all rows
of a (perturbation, context) pair, a group lasso (`lasso_groups="pair"`); the
penalty is then $\lambda\sqrt{n_g}\sum_k\|\mathbf Z_g[:,k]\|_2$ on each
program's column of activities over the pair's $n_g$ rows
(`docs/notes/group_lasso/group_lasso.pdf`). It makes a row's activities
depend on the other rows of its pair, so the discriminator no longer judges
rows independently; it is kept only for comparison.

### Architecture

There are three learned parts and no encoder: the loadings $\mathbf B$, a
label network, and a per-component head (`cryptex.train.Cryptex`).

```{raw} html
<div class="ex-figure">
<svg viewBox="0 0 780 330" width="780" height="330" role="img" aria-label="Forward pass: x is masked; its activities z are found by a lasso fitted to the row alone; four statistics of z are weighted by label coefficients theta, a label-only normaliser is subtracted, and the sum is the logit; z times B reconstructs x.">
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>
    <marker id="aB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahB"/></marker>
  </defs>
  <rect class="box" x="16" y="36" width="104" height="52" rx="4"/>
  <text x="68" y="59" text-anchor="middle" font-size="15" font-weight="600">x</text>
  <text x="68" y="77" text-anchor="middle" class="sub">logFC · G genes</text>
  <rect class="box" x="150" y="36" width="130" height="52" rx="4"/>
  <text x="215" y="59" text-anchor="middle" font-size="13.5">mask on-target</text>
  <text x="215" y="77" text-anchor="middle" class="sub">x̃ = m ⊙ x</text>
  <rect class="box" x="310" y="36" width="140" height="52" rx="4"/>
  <text x="380" y="59" text-anchor="middle" font-size="13.5">z: lasso</text>
  <text x="380" y="77" text-anchor="middle" class="sub">each row on its own</text>
  <rect class="box" x="480" y="36" width="140" height="52" rx="4"/>
  <text x="550" y="59" text-anchor="middle" font-size="13.5">q<tspan font-size="10" dy="3">j</tspan><tspan dy="-3">(z</tspan><tspan font-size="10" dy="3">k</tspan><tspan dy="-3">)</tspan></text>
  <text x="550" y="77" text-anchor="middle" class="sub">4 statistics − c(u) · d</text>
  <circle class="box" cx="668" cy="62" r="20"/>
  <text x="668" y="67" text-anchor="middle" font-size="15">Σ</text>
  <rect class="box" x="706" y="36" width="62" height="52" rx="4"/>
  <text x="737" y="59" text-anchor="middle" font-size="13.5">s(x,u)</text>
  <text x="737" y="77" text-anchor="middle" class="sub">logit</text>
  <path class="ln" d="M120 62 H148" marker-end="url(#a)"/>
  <path class="ln" d="M280 62 H308" marker-end="url(#a)"/>
  <path class="ln" d="M450 62 H478" marker-end="url(#a)"/>
  <path class="ln" d="M620 62 H646" marker-end="url(#a)"/>
  <path class="ln" d="M688 62 H704" marker-end="url(#a)"/>
  <rect class="boxB" x="310" y="150" width="140" height="52" rx="4"/>
  <text x="380" y="173" text-anchor="middle" font-size="14" font-weight="600">B</text>
  <text x="380" y="191" text-anchor="middle" class="sub">d × G · learned</text>
  <path class="lnB" d="M380 150 V90" marker-end="url(#aB)"/>
  <text x="388" y="124" class="sub">B = A V · unit rows</text>
  <rect class="box" x="150" y="150" width="130" height="52" rx="4"/>
  <text x="215" y="173" text-anchor="middle" font-size="13.5">x̂ = z B</text>
  <text x="215" y="191" text-anchor="middle" class="sub">no bias term</text>
  <path class="lnB" d="M310 176 H282" marker-end="url(#aB)"/>
  <path class="ln" d="M322 88 L262 148" marker-end="url(#a)"/>
  <rect class="loss" x="16" y="150" width="104" height="52" rx="4"/>
  <text x="68" y="173" text-anchor="middle" font-size="13.5">L_lasso</text>
  <text x="68" y="191" text-anchor="middle" class="sub">error + λ · penalty</text>
  <path class="ln" d="M150 176 H122" marker-end="url(#a)"/>
  <path class="ln" d="M68 88 V148" stroke-dasharray="3 3" marker-end="url(#a)"/>
  <rect class="box" x="310" y="256" width="140" height="52" rx="4"/>
  <text x="380" y="279" text-anchor="middle" font-size="13.5">u = (p, c)</text>
  <text x="380" y="297" text-anchor="middle" class="sub">own or permuted</text>
  <rect class="box" x="480" y="256" width="140" height="52" rx="4"/>
  <text x="550" y="279" text-anchor="middle" font-size="13.5">[e_p ; e_c] → MLP</text>
  <text x="550" y="297" text-anchor="middle" class="sub">θ(u) · d×4, c(u) · d</text>
  <path class="ln" d="M450 282 H478" marker-end="url(#a)"/>
  <path class="ln" d="M620 282 H668 V84" marker-end="url(#a)"/>
  <text x="660" y="230" text-anchor="end" class="sub">θ ⊙ q(z)</text>
  <rect class="loss" x="706" y="150" width="62" height="52" rx="4"/>
  <text x="737" y="173" text-anchor="middle" font-size="13.5">L_disc</text>
  <text x="737" y="191" text-anchor="middle" class="sub">BCE</text>
  <path class="ln" d="M737 88 V148" marker-end="url(#a)"/>
</svg>
</div>
<p class="ex-caption">One forward pass. Blue marks B and the two maps derived from it; red marks the loss terms. The label u reaches the score only through θ, which multiplies per-factor statistics before the sum, and the label-only term c.</p>
```

#### Loadings: one map, no encoder

$\mathbf B = \mathbf A\mathbf V$ is the only factor → gene map. Activities are
not predicted by a network; they are fitted to the row through $\mathbf B$ (the
lasso above). A separate encoder would give two factor → gene maps that
can disagree about what factor $k$ means, and encoder weights are filters, not
patterns (Haufe et al. 2014): a gene can get a large weight precisely to cancel
it. Fitting the activities through $\mathbf B$ makes $\mathbf B$ a pattern by
construction. Its rows are kept at unit length after every optimiser step.

The reconstruction is $\hat{\mathbf x} = \mathbf z\mathbf B$ with no bias,
since $\mathbf x$ is already a difference from controls. The squared error
skips the masked entry and is divided by the number of retained genes, so its
scale does not depend on $G$.

#### Label network: where all the interaction lives

The label network maps $(p, c)$ to the head's coefficients. It concatenates a
perturbation embedding $e_p$ and a context embedding $e_c$ and passes them
through `Dropout → Linear → ReLU → Dropout → Linear(·, d·5)`, with weight decay
on the label network; dropout and weight decay keep it from memorising the
training pairs. The context embedding can be much smaller than the perturbation
embedding: a screen usually has far fewer contexts than perturbations ($C$
contexts need at most $C-1$ dimensions), and the PCA of the contexts' control
cells (below) shows how many dimensions carry their differences.

**Starting perturbation embeddings: how the target gene responds.** A
perturbation's embedding starts from how its *target gene* behaves as a
response gene when other genes are perturbed. Take the mean of each training
pair's rows, keep the columns of the target genes, and transpose: one row per
target gene, one column per training pair. The entries where the pair's own
perturbation targets that gene are set to zero, since the knockdown itself
would dominate. A truncated SVD of this matrix to the embedding size gives each
target gene's starting embedding, rescaled to the size of a random start. Genes
affected alike by many perturbations tend to sit in the same pathway, so their
knockouts start close together. Only training pairs are used, so validation and
test pairs never shape an embedding. Perturbations whose target is not among the
response genes start at random. (`cryptex.train.response_embeddings`.)

**Starting context embeddings: the control cells.** A context's embedding
starts from the mean expression of its control (non-targeting) cells: the
control cells each context's logFC were computed against, averaged per
gene over the response genes. PCA of the contexts × genes matrix (each gene
centred across the contexts, not relative to a reference context) gives each
context's scores on the top components (as many as the embedding size), rescaled to the size of a random start. The
logFC are measured against each context's own controls, so the context's direct
effect on expression is removed from the data; what the control cells add is
the cell state the context creates, which shapes how perturbations act.
(`cryptex.train.control_mean_embeddings`, computed by
`scripts/context_control_means.py`.)

Both kinds of embedding are only starting points and are trained further.

The factorisation matters. A free embedding per observed pair would let the
optimal discriminator become a lookup table ("answer real if and only if
$\mathbf x$ is the vector stored for $u$"), reachable without any notion of
components. Sharing $e_p$ across contexts and $e_c$ across perturbations cuts
the parameters from $n_{\text{pairs}} \cdot d$ to $(n_p + n_c) \cdot d$ and
makes memorisation impossible. It is also what lets a held-out pair, whose
perturbation and context were both seen but never together, be scored at all.

#### Per-component head

For a row with activities $\mathbf z$ and a candidate label $u = (p, c)$,

$$
s(\mathbf x, u) \;=\; b \;+\; \sum_{k=1}^{d}\Big[\sum_{j=1}^{4} \theta_{kj}(u)\, q_j(z_k) \;-\; c_k(u)\Big],
\qquad
\Pr(\text{true label} \mid \mathbf x, u) = \sigma\big(s(\mathbf x, u)\big),
$$

with the statistics $q = (z,\ z^2,\ |z|,\ \tanh z)$ applied coordinate-wise.
The label network gives, per program, the four weights $\theta_{k\cdot}(u)$ and
the term $c_k(u)$. No term ever multiplies $z_k$ by $z_{k'}$; the head's own
parameter is the bias $b$.

**What the head models.** For each program, $\exp\big(\sum_j \theta_{kj}(u)\,q_j(z_k) - c_k(u)\big)$
is a distribution of that program's activity under the label: $z$ and $z^2$ set
its centre and width, $|z|$ lets it be sparse (mostly zero, sometimes large),
$\tanh z$ lets it be one-sided. A row scores high when each of its activities
falls where its label's distribution puts mass.

**Independent given the label, free across labels.** Because the score is a sum
over programs, with no term combining two programs, it is the log-likelihood of
a model in which the programs vary **independently given the label**: for one
(perturbation, context), how far a row's activity on program $k$ falls from that
label's typical value says nothing about program $l$. This is an assumption the
head encodes, not a constraint imposed on the activities; axes on which it holds
make the discrimination easier, which is how the head prefers them. Across
labels the programs are not required to be independent: the label network can
make a perturbation move several programs together, so over the whole screen
two programs' activities may be strongly correlated. This differs from ICA, which
makes the activities independent over all samples (marginal independence). Since
the assumption concerns the activities, nothing requires the programs' gene
loadings to be orthogonal: independent programs may share genes and point in
similar directions in gene space, which is why $\mathbf A$ is a general
invertible matrix rather than a rotation. The lasso adds a related property:
each response uses few programs, so each program's activity is exactly zero
for most responses and large for a few.

**Why the label-only term $c_k(u)$.** It plays the role of the distribution's
normaliser, $\log Z_k(u)$: the amount each label must subtract so that its
distribution has total mass one. Without it, a label whose coefficients describe
a large effect would score high for any row that points its way, whether or not
the row is close to that effect. The classifier learns $c_k(u)$ directly; only
$\sum_k c_k(u)$ enters the score.

**Why this form.** With label swaps as the false examples, the best possible
classifier has the logit $\log p(\mathbf x, u) / \big(p(\mathbf x)\,p(u)\big)$.
If the programs are independent given the label and each has an
exponential-family distribution in these statistics, that logit is exactly
$\sum_k \big[\boldsymbol\theta_k(u)\cdot\mathbf q(z_k) - \log Z_k(u)\big]$ plus
terms that do not depend on the label: the head's form, with $c_k = \log Z_k$.
This is generalised contrastive learning (Hyvärinen, Sasaki & Turner 2019; the
same assumption as time-contrastive learning and iVAE). Separability in the
programs is not preserved under mixing, and the non-quadratic statistics
($|z|$, $\tanh z$) are not functions of the length of $\mathbf z$, so a rotated
set of axes cannot reproduce the same scores. A basis of only `square` is
refused, because $\sum_k z_k^2$ is rotation invariant.

The theorem's conditions (noiseless mixing, enough label-driven variation of the
programs' distributions) hold only approximately in real screens. When most of
the label signal is in the programs' means, the head alone constrains the
rotation only weakly; the starting axes and the lasso supply the rest.

#### Shapes

| Tensor | Shape | Where | Note |
|---|---|---|---|
| `x` | [batch, G] | input | unscaled shrunken logFC |
| `target_col` | [batch] | input | column to mask, −1 for none |
| `B` | [d, G] | loadings | $\mathbf A\mathbf V$, unit rows; the interpretable output |
| `z` | [batch, d] | lasso | zero for unused programs |
| `q(z)` | [batch, d, 4] | head | four statistics |
| `e_p` | [batch, $d_p$] | label network | shared perturbation embedding |
| `e_c` | [batch, $d_c$] | label network | shared context embedding |
| `θ`, `c` | [batch, d, 4], [batch, d] | label network | label-dependent weights and normaliser |
| `s` | [batch] | head | true/false logit |

### Objective

$$
\mathcal L \;=\; \mathcal L_{\text{disc}} \;+\; \alpha\,\mathcal L_{\text{lasso}}
\;+\; \gamma\,\mathcal L_{\text{genes}} \;+\; \mu\,\mathcal L_{\text{distinct}} .
$$

**Discrimination.**
$\mathcal L_{\text{disc}} = \mathrm{BCE}\big(s(\mathbf x, u), 1\big) + \mathrm{BCE}\big(s(\mathbf x, u^{-}), 0\big)$,
each term averaged over its own rows, on the lasso activities. A false
example keeps the observed vector and swaps the label, so the distribution of
$\mathbf x$ is the same for true and false examples and the only way to win is
to model how $\mathbf x$ depends on $u$.

**The lasso objective.**

$$
\mathcal L_{\text{lasso}} = \frac1n\sum_{\text{rows } r}\frac{\|\mathbf m_r\odot(\mathbf x_r - \mathbf z_r\mathbf B)\|^2}{G_r}
\;+\; \frac{2\lambda}{nG}\sum_{\text{rows } r}\sum_k|z_{rk}| ,
$$

over the batch's $n$ measured rows, with the lasso activities; $G_r$ is the
number of genes row $r$ keeps ($G$, or $G-1$ when its target gene is masked).
It is the per-row objective above, summed over the batch's rows and divided by
$nG/2$ (up to that one masked gene), so it is on a per-row, per-gene scale.
$\lambda$ appears twice: in the solve (which programs a row uses) and here (so
the gradient on $\mathbf A$ makes axes on which responses need fewer programs
cheaper). Every row weighs the same.

**The weight $\alpha$.** The discrimination term is of order 1, while the
reconstruction error per gene is of order the mean squared logFC, typically
far below 1. So $\alpha$ must be large for the two terms to interact; it is
tuned on validation.

**Few genes per program.** For each program $\mathbf b_k$ (a row of $\mathbf B$),

$$
\mathcal L_{\text{genes}} = \sum_{k=1}^{d} \frac{\|\mathbf b_k\|_1}{\|\mathbf b_k\|_2\,\sqrt G} .
$$

The ratio is scale-free: it is 1 for a program that weights all genes equally
and $1/\sqrt G$ for a program on a single gene. Minimising it moves weight off
the many small loadings onto a few large ones, so each program's loadings
become heavy-tailed: mostly near zero, a few large. Absolute values are used,
so a program may combine up- and down-regulated genes. No number of genes is
imposed; how concentrated the programs become is set by $\gamma$, against the
other terms.

**Distinct programs.** A sparsity reward alone is cheapest to satisfy by
letting several programs share the same few large-response genes, which makes
them near-copies. A hinge forbids that:

$$
\mathcal L_{\text{distinct}} = \sum_{k<l} \max\big(0,\; |\cos(\mathbf b_k, \mathbf b_l)| - \tau\big)^2 ,
$$

with the cosine taken over the full loading vectors. It is zero as long as two
programs are less similar than $\tau$, i.e. the angle between them (between the
lines, since a program and its negative are the same program) is at least
$\arccos\tau$; above that it grows quickly, and a large $\mu$ makes it act as a
limit. Below the limit, programs may share genes and overlap freely: real
programs are not orthogonal, and independent programs (in the sense of the
head) can point in similar directions in gene space.

Both terms depend only on $\mathbf B$, not on the data rows; they are computed
once per step and act on $\mathbf A$. Distinctness is best checked in two
ways: the cosine between the programs' loading vectors, and the overlap
(Jaccard index) of their selected gene sets.

### Negatives

The false examples are measured rows under a wrong label, one per measured row,
drawn afresh every epoch:

| Strategy | Share | Swap | Forces |
|---|---|---|---|
| other perturbation, same context | 0.5 | $p$, within $c$ | perturbation identity |
| same perturbation, other context | 0.5 | $c$, within $p$ | the interaction |

Contexts are symmetric: no strategy assumes a reference arm such as vehicle.
With one context only the first strategy exists, and it suffices.

- **Only observed pairs.** The sampler is built from training rows only, and
  every drawn label is a pair observed among them, so a validation or test pair
  is never named, not even as a negative. If one axis has no observed
  alternative the other axis is swapped; if neither has, it is an error rather
  than a silent leak.
- **Balanced.** Each measured row is one true and one false example, so every
  pair contributes equal numbers of both.
- **Drawn in proportion to rows.** The wrong label is drawn by picking a random
  row among the candidates (same context and another perturbation, or the same
  perturbation and another context) and taking its label, so a label with many
  rows is drawn more often. A pair with more cells has more pseudobulk rows and
  so appears more often as a true label; drawn this way, it also appears about
  as often as a false label. Otherwise the label alone would carry information,
  since labels with many rows would be true more often than false, and the
  label-only term $c_k(u)$ could learn which perturbations have many cells
  instead of anything about the responses. This matters most for rows that use
  no program, whose activities are all zero and whose score is the label term
  alone.
- **Only label swaps.** This is the setting in which the best classifier is the
  density ratio above, so the head's form is the right one. Artificial
  negatives (for example rows with their genes shuffled) would ask a different
  question (is this a plausible response at all?), whose optimal classifier is
  not of the head's form.

### Training loop

`fit()` takes the rows, their integer-encoded perturbation and context, the
on-target column and train/validation masks. First it
computes $\mathbf V$ from the measured training rows, the starting axes
$\mathbf A_0$ by joint diagonalisation, and the starting embeddings. Then, per
step:

1. Draw a batch of rows in random order; the sampler returns a wrong label for
   each measured row.
2. Solve the lasso for every row of the batch with the current $\mathbf B$,
   with the on-target mask.
3. Score true labels $s(\mathbf z, p, c)$ and false labels
   $s(\mathbf z, p^*, c^*)$; the same $\mathbf z$ serves both.
4. Compute $\mathcal L_{\text{disc}}$, $\mathcal L_{\text{lasso}}$ and the two gene-axis
   terms from the current $\mathbf B$; step with
   Adam (with gradient-norm clipping; weight decay on the label network), updating $\mathbf A$, the label network and the head bias.
5. Rescale the rows of $\mathbf B = \mathbf A\mathbf V$ to unit length.

At regular intervals the validation rows are scored, and every loss term is recorded
for training and validation.

### Splits and model selection

**Three partitions, split by pair.** All rows of a (perturbation, context) pair
come from the same cells, so they always move together.

| Partition | Drawn | Used for |
|---|---|---|
| test | a fraction of pairs, drawn first, from metadata alone; every perturbation keeps a train/val pair | read once, at the end |
| validation | a fraction of each perturbation's train/val pairs, at least one kept in train | choosing the epoch and the settings |
| train | the rest | fitting |

A held-out pair is an unseen combination of a seen perturbation and a seen
context: the model has to supply the interaction.

**Choosing the epoch.** Every model in a comparison trains for the same fixed
number of epochs, and the kept state
is the epoch with the highest **validation ROC AUC**: how well the scores rank
validation rows under their own label above the same rows under a wrong label.
The wrong labels of validation rows are drawn the same way as in training, but
from the validation pairs, so validation never names a training pair either.
AUC depends only on the ranking, so it compares models whose scores are on
different scales.

**Choosing settings.** The hyperparameters ($d$, $\lambda$, $\alpha$, the
embedding sizes and the label-network regularisation) are tuned for each dataset.
Models are compared on validation only, each over several random seeds:
validation AUC, how well the programs agree between seeds, and how well they
agree between models trained on two disjoint halves of the perturbations (both
in one shared $\mathbf V$), the most direct test that the programs are set by
the data rather than by the run.
Differences smaller than the spread across seeds are not differences.

**What is saved.** Each model's weights at the selected epoch and at the last
epoch, its configuration, the label encoding, the full history of every loss
term, and its validation scores, so any model can be reloaded and re-scored.

**The test read.** The chosen configuration's test metrics are read once, and
the read is logged.

### Evaluation

| Metric | Tests |
|---|---|
| held-out AUC | Can the model tell an unseen $(p, c)$ from a wrong label? A lookup table scores 0.5. |
| programs per row | How many programs the lasso uses for a held-out row. |
| reconstruction | Masked squared error per gene of $\mathbf z\mathbf B$, with the lasso activities; `cryptex.evaluation.reconstruction` also gives the fraction of squared norm recovered by $\operatorname{span}(\mathbf B)$ (least-squares activities). |
| span residual $r$ | $\|\mathbf m \odot (\mathbf x - \mathbf x\mathbf B^{+}\mathbf B)\| / \|\mathbf m \odot \mathbf x\|$: do held-out rows need programs outside $\operatorname{span}(\mathbf B)$? |
| seed agreement | Matched $|\cos|$ between the programs of models trained with different seeds or on different data (Hungarian matching, no centring; `cryptex.evaluation.stability`), against the agreement of two random sets of axes in $\mathbf V$. |

### Reading the factors

**Loadings.** Row $\mathbf B_k$ is program $k$, already in logFC units; no
back-transform is needed.

**Sign and scale.** Both are free per factor: $(-z_k)(-\mathbf B_k) =
z_k\mathbf B_k$. Signs are anchored by a reference perturbation whose direction
is known or, failing that, by making the largest-magnitude loading positive.
Gene lists for enrichment report the two tails separately.

**Activations.** $z_k$ is the change in program $k$ relative to unperturbed
cells of the same context: positive is up-regulation, negative is
down-regulation, and exactly zero means the pair does not use program $k$.

**Effect decomposition.**
$z_{pc} = \bar z + \mu_p + \gamma_c + \delta_{pc}$ separates the perturbation
main effect, the context main effect and the interaction. $\delta_{pc}$ is
epistasis on the log scale, identified by the multiple rows per pair.

**Responsiveness.** $a_{kc} = \operatorname{Var}_p[z_k(p, c)]$ says how
modulable program $k$ is in context $c$: large when the panel engages it, near
zero when the context has locked the program or when nothing engages it.

### Configuration

The model is configured through `TrainConfig` (and the matching flags of
`scripts/sweep_grid.py`). The structural choices described above are the
defaults; the hyperparameters are dataset-specific and should be tuned on
validation.

**Structure**

| Field | Role |
|---|---|
| `subspace` / `subspace_method` | `fixed` / `pca`: $\mathbf B = \mathbf A\mathbf V$ with $\mathbf V$ the PCA of the pair means; `free` learns $\mathbf B$ directly |
| `axes_init` | `jd`: start $\mathbf A$ at the joint-diagonalisation axes; `random`: a random rotation per seed |
| `coding` | `group_lasso`: activities by the lasso solver (below); `ls` for plain least squares |
| `lasso_groups` | `row`: a lasso per row (each row one data point); `pair`: one selection shared by a pair's rows (group lasso, for comparison) |
| `basis` | the head's statistics (`linear`, `square`, `abs`, `tanh`); must keep a non-quadratic one |
| `negative_draw` | `rows`: wrong labels in proportion to their rows (above); `labels`: uniformly over the observed alternatives |
| `pert_embedding_init` | `response`: start from the target gene's responses; contexts from the control means (`--context-embedding-init control-means`) |
| `no_mask` | ablation: no on-target mask |

**Hyperparameters to tune**

| Field | Role |
|---|---|
| `n_factors` | $d$, the number of programs |
| `lasso_lambda` | $\lambda$, how many programs a row uses |
| `alpha` | $\alpha$, weight of $\mathcal L_{\text{lasso}}$ against $\mathcal L_{\text{disc}}$ |
| `embedding_dim` / `context_embedding_dim` / `label_hidden` | label-network sizes |
| `label_dropout` / `label_weight_decay` | label-network regularisation |
| `epochs` / `batch_size` / `lr` / `eval_every` | training length, batch size, learning rate, validation interval |
| `lasso_steps` / `ridge` | lasso (FISTA) iterations; conditioning of the least-squares starting point |
| `gene_sparsity` | $\gamma$, how concentrated the programs' loadings become |
| `distinct_tau` / `distinct_weight` | $\tau$, the largest allowed $|\cos|$ between two programs (minimum angle $\arccos\tau$); $\mu$, how strictly it is enforced |

### Module map

| Path | Contents |
|---|---|
| `cryptex/de/` | Welch statistics → `ashr` → merge → pivot; full-data and disjoint pseudobulk rows; matrix assembly |
| `cryptex/data/` | `noise.py` the subspace $\mathbf V$ (PCA of pair means; signal-to-noise variants); `splits.py` test pairs, validation, the low-cell filter, gene lists |
| `cryptex/models/` | `loadings.py` $\mathbf B = \mathbf A\mathbf V$ and projections; `group_lasso.py` the lasso solver (per row, or per pair as a group lasso); `heads.py` the per-component head; `label_net.py` |
| `cryptex/objectives/` | `contrastive.py` label-swap sampler and BCE; `reconstruction.py` the masked squared error |
| `cryptex/train.py` | `TrainConfig`, `Cryptex`, `fit`, `evaluate`, `jd_axes`, embedding starts, `load_model` |
| `cryptex/interpret/` | loadings, sign anchoring, effect decomposition, enrichment export |
| `cryptex/evaluation/` | held-out reconstruction, span residual, cross-seed stability |
| `baselines/` | PCA, linear ICA, context joint diagonalisation |
| `scripts/` | build the split (`build_split.py`), train models (`sweep_grid.py`; `fit_model.py` for one), read test once (`read_test.py`) |
