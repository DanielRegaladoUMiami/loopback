# loopback: A Reproducible Two-Tower Reference Implementation for Music Retrieval

**Daniel Regalado**
University of Miami · `dxr1491@miami.edu`
[github.com/DanielRegaladoUMiami/loopback](https://github.com/DanielRegaladoUMiami/loopback)

---

## Abstract

We present **loopback**, an open, from-scratch PyTorch reference implementation of
two-tower neural retrieval for music recommendation. The system embeds users and
tracks into a shared 128-dimensional space with separate MLP towers, and is
trained with a symmetric InfoNCE objective using in-batch negatives and a
CLIP-style learnable temperature. Trained on the Last.fm 1K-users dataset
(19.1M listening events, 992 users, 1.5M tracks), the model attains
recall@10 = 0.0708, recall@50 = 0.2172, and recall@100 = 0.3140 against the full
1.5M-track catalog on a held-out temporal split — four orders of magnitude above
a random baseline. The training run completes in roughly 27 minutes on a single
Apple-silicon GPU (MPS). We release the processed dataset, trained checkpoint,
and a live FAISS-backed demo on the Hugging Face Hub. The contribution is not a
new architecture but a *readable, reproducible, mathematically explicit*
reference for a model class that is widely deployed but rarely written up at
this granularity outside proprietary settings.

## 1. Introduction

Two-tower retrievers are the dominant architecture for large-scale candidate
generation: YouTube, Google Play, Spotify, and Pinterest have all published
variants of the same idea — encode a query and a candidate into the same
vector space, train with a contrastive objective, and serve with approximate
nearest neighbor search. Despite the ubiquity of the pattern, public
implementations tend to fall into two buckets: (i) toy notebooks on MovieLens
that skip the loss math and never index the full catalog, or (ii) industrial
codebases that bury the model under serving infrastructure.

We target the gap in between. We pick a domain (music) where the long-tail
catalog is non-trivial (1.5M items), use a public dataset of real implicit
feedback (Last.fm 1K), and write the model, loss, and evaluation tightly
enough that the entire training loop fits in under 80 lines of PyTorch.
The objective is pedagogical clarity backed by real numbers — every design
choice (in-batch negatives, learnable temperature, L2-normalized embeddings,
FAISS inner-product index) is stated explicitly and tied to the math.

## 2. Related Work

**Matrix factorization.** Implicit-feedback CF was framed as weighted
matrix factorization by Hu et al. [Hu et al. 2008] (ALS) and as pairwise
ranking by Rendle et al. [Rendle et al. 2009] (BPR). These methods learn a
single inner-product space (`u · t`) and remain strong baselines, but they
do not extend cleanly to side information or sequence inputs.

**Two-tower and dual-encoder retrieval.** Covington et al. [Covington et
al. 2016] introduced the YouTube DNN candidate generator, framing
recommendation as extreme multiclass classification with sampled softmax.
Yi et al. [Yi et al. 2019] formalized the corrected-in-batch-softmax
two-tower at Google and showed how to debias popularity in the negative
distribution. CLIP [Radford et al. 2021] uses the same symmetric InfoNCE
objective with a learnable temperature for image-text retrieval; we
borrow that parameterization directly.

**Music recommendation.** Spotify's Music Recommendations work and
sequential models such as SASRec [Kang & McAuley 2018] go beyond the
bag-of-interactions assumption and condition on the user's listening
history as a sequence. loopback intentionally stops short of this — the
user side is a lookup table — so the in-batch InfoNCE result is not
confounded by a sequence encoder. Audio-content towers (e.g., the
Spotify musicnn line of work) are an obvious extension that we leave
to ongoing work.

## 3. Method

### 3.1 Architecture

```
              ┌───────────────────────────┐
   user_id ──▶│  Embedding(n_users, 64)   │──▶ MLP(64→256→128) ──▶ L2-norm ──▶ u  (B,128)
              └───────────────────────────┘
                                                                                │
                                                                       u · tᵀ   │ (B,B)
                                                                                │
              ┌───────────────────────────┐                                     │
  track_id ──▶│  Embedding(n_tracks, 64)  │─┐                                   │
              └───────────────────────────┘ │                                   │
                                            ├▶ concat ─▶ MLP(128→256→128) ─▶ L2 ─▶ t  (B,128)
              ┌───────────────────────────┐ │
 artist_id ──▶│  Embedding(n_artists, 64) │─┘
              └───────────────────────────┘
```

Each tower is a single hidden-layer MLP with GELU and dropout 0.1. The
track tower concatenates a track-ID embedding with its artist-ID embedding
before projection, giving the model an explicit popularity-and-style
backoff signal for long-tail tracks that share artists with frequent ones.
Both tower outputs are L2-normalized so that the inner product `u · t` is
cosine similarity in `[-1, 1]`. A single learnable scalar `log τ` scales the
logits before softmax; we initialize `log τ = 0`, so `τ = 1` at the start of
training.

### 3.2 Contrastive Objective

Given a minibatch of `B` observed `(user, track)` pairs, the towers produce
`u ∈ ℝ^{B×128}` and `t ∈ ℝ^{B×128}`. We form the similarity matrix

$$
S_{ij} \;=\; (u_i \cdot t_j) \cdot \exp(\log \tau), \qquad S \in \mathbb{R}^{B\times B}.
$$

The diagonal `S_{ii}` is the score of each user's true positive. Every
off-diagonal entry `S_{ij}` with `i ≠ j` is treated as an implicit negative:
some *other* user's positive track, sampled (for free) from the empirical
data distribution.

The user→track loss is a row-wise softmax cross-entropy:

$$
\mathcal{L}_{u\rightarrow t} \;=\; -\,\frac{1}{B}\sum_{i=1}^{B}\; \log
\frac{\exp(S_{ii})}{\sum_{j=1}^{B}\exp(S_{ij})}.
$$

The track→user loss is the column-wise analogue, and the total loss is the
symmetric average — the CLIP objective applied to recommendation:

$$
\mathcal{L} \;=\; \tfrac{1}{2}\bigl(\mathcal{L}_{u\rightarrow t} + \mathcal{L}_{t\rightarrow u}\bigr).
$$

**Why in-batch negatives work.** The full softmax over a 1.5M-track catalog
is intractable. In-batch sampling replaces the partition function with a
Monte Carlo estimate over `B − 1` negatives drawn from the popularity-biased
data distribution. This is exactly the sampled-softmax estimator of
[Covington et al. 2016] without the log-Q correction; we accept the
resulting popularity bias as the price of simplicity, and document the
consequence in §6.

**What the temperature does.** `τ` controls the peakedness of the softmax.
A large `τ` (small effective temperature `1/τ`) makes the loss sharply
discriminative — gradients concentrate on the hardest in-batch negative —
while a small `τ` flattens the distribution and slows learning. Making `τ`
learnable lets the model anneal itself: in our run `exp(log τ)` rises from
1.0 at initialization to approximately 28 by the end of epoch 3, mirroring
the range reported for CLIP.

**A sanity check on the loss scale.** With `B = 4096` and a random model,
the softmax denominator has `B` roughly equal terms, so the expected loss
is `ln B ≈ 8.32`. Our final training loss is `≈ 5.6`, well below the random
floor, indicating that the model is recovering signal rather than memorizing
batch order.

## 4. Experiments

### 4.1 Dataset

We use the Last.fm 1K-users dataset [Celma 2010], a public log of scrobbles
from 2005–2009 with schema
`(user_id, timestamp, artist_mbid, artist_name, track_mbid, track_name)`.
After deduplication and ID remapping:

| | count |
|---|---:|
| Users | 992 |
| Unique tracks | 1{,}503{,}123 |
| Unique artists | 173{,}920 |
| Interactions (total) | 19{,}098{,}862 |
| Train / Val / Test | 15.3M / 1.9M / 1.9M |

Splits are **temporal per user**: for each user we sort interactions by
timestamp and take the earliest 80% as train, the next 10% as val, and the
final 10% as test. This avoids the well-known leakage that random splits
introduce in sequential domains.

### 4.2 Training Configuration

| Hyperparameter | Value |
|---|---|
| Output embedding dim | 128 |
| Internal ID embedding dim | 64 |
| MLP hidden | 256 |
| Dropout | 0.1 |
| Batch size | 4096 |
| Optimizer | AdamW (`lr = 1e-3`, `wd = 1e-5`) |
| Loss | symmetric InfoNCE |
| Temperature | learnable, `log τ` init 0 |
| Epochs | 3 |
| Hardware | Apple-silicon GPU (PyTorch MPS) |
| Wall time | ≈ 9 min/epoch |

### 4.3 Evaluation Protocol

At evaluation time we embed all 1.5M tracks with the trained track tower
and build a `faiss.IndexFlatIP` over the L2-normalized vectors (inner
product = cosine on the unit sphere). For each test user we query the user
tower, retrieve the top `K + |history|` neighbors, **filter out tracks the
user already heard during training**, and report recall@K against the test
positives. We evaluate over the 847 test users that also appear in the
training split. The full 1.5M-track catalog is used as the candidate set —
we do not subsample negatives or restrict to popular tracks at evaluation.

### 4.4 Results

| Metric | loopback | Random baseline |
|---|---:|---:|
| Recall@10  | **0.0708** | 6.7 × 10⁻⁶ |
| Recall@50  | **0.2172** | 3.3 × 10⁻⁵ |
| Recall@100 | **0.3140** | 6.7 × 10⁻⁵ |

The random baseline is `K / N_tracks` and serves only to anchor scale:
loopback is roughly four orders of magnitude above chance. Recall grows
sub-linearly in `K`, as expected for a long-tail catalog.

### 4.5 Observations

- The learnable temperature converged to `exp(log τ) ≈ 28`, close to the
  CLIP regime, without any explicit clamping.
- Training loss curves are monotonically decreasing across all three epochs;
  no overfitting signal was observed at this depth and dataset size.
- FAISS index build takes < 30 s for 1.5M vectors at dim 128; per-user
  top-100 retrieval is sub-millisecond on CPU.

## 5. Limitations

We are explicit about what loopback does *not* do, in roughly decreasing
order of impact on quality.

1. **In-batch negatives only.** We do not mine hard negatives nor apply the
   log-Q popularity correction of [Yi et al. 2019]. The model is therefore
   biased toward popular tracks at retrieval time. Hard-negative mining and
   log-Q correction are ongoing.
2. **No content features.** The track tower uses only `(track_id, artist_id)`
   embeddings, so the model is pure collaborative filtering and cannot
   generalize to cold-start tracks. Audio embeddings (e.g., from a CNN over
   log-mel spectrograms, or a pre-trained MERT/CLAP encoder) are an obvious
   next axis.
3. **No user-sequence modeling.** The user tower is a single ID lookup,
   which discards temporal order entirely. Replacing it with a SASRec- or
   BERT4Rec-style sequence encoder over recent listens is a natural
   upgrade — and would let the model condition on session context at serve
   time.
4. **Dataset age.** Last.fm 1K covers 2005–2009. Listening patterns,
   catalog composition, and the popularity distribution have shifted
   considerably since; results here should be read as a methodology
   demonstration, not a contemporary benchmark.
5. **Popularity-skewed evaluation.** Temporal splits with implicit
   feedback inherit the popularity bias of the training distribution. Tail
   tracks appear rarely in both train and test, so recall on them is
   poorly estimated. Stratified-by-popularity evaluation is ongoing.

## 6. Reproducibility

All code, the processed dataset, the trained checkpoint, and a live demo
are public.

- **Code:** [github.com/DanielRegaladoUMiami/loopback](https://github.com/DanielRegaladoUMiami/loopback) (Apache 2.0)
- **Dataset:** [huggingface.co/datasets/DanielRegaladoCardoso/lastfm-1k-twotower](https://huggingface.co/datasets/DanielRegaladoCardoso/lastfm-1k-twotower)
- **Model:** [huggingface.co/DanielRegaladoCardoso/loopback-twotower](https://huggingface.co/DanielRegaladoCardoso/loopback-twotower)
- **Demo:** [huggingface.co/spaces/DanielRegaladoCardoso/loopback](https://huggingface.co/spaces/DanielRegaladoCardoso/loopback)

To reproduce end-to-end:

```bash
git clone https://github.com/DanielRegaladoUMiami/loopback && cd loopback
uv sync
uv run python -m loopback.data prepare    # download + ID-map Last.fm 1K
uv run python -m loopback.train           # 3 epochs, ~27 min on MPS
uv run python -m loopback.eval            # recall@10, @50, @100
```

The training script is deterministic given a fixed seed and the published
processed parquet files; the numbers in §4.4 should reproduce within
±0.005 absolute recall across runs.

## References

- Celma, Ò. *Music Recommendation and Discovery in the Long Tail*, Springer, 2010.
- Covington, P., Adams, J., Sargin, E. *Deep Neural Networks for YouTube Recommendations*. RecSys 2016.
- Hu, Y., Koren, Y., Volinsky, C. *Collaborative Filtering for Implicit Feedback Datasets*. ICDM 2008.
- Johnson, J., Douze, M., Jégou, H. *Billion-scale similarity search with GPUs* (FAISS). IEEE Trans. Big Data 2019.
- Kang, W.-C., McAuley, J. *Self-Attentive Sequential Recommendation* (SASRec). ICDM 2018.
- Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
- Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L. *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI 2009.
- van den Oord, A., Li, Y., Vinyals, O. *Representation Learning with Contrastive Predictive Coding* (InfoNCE). arXiv:1807.03748, 2018.
- Yi, X. et al. *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*. RecSys 2019.
