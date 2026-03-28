# Semantic KV with Straight-Through Greedy Selection

This note describes the implementation added on `trial-mar28`.

It is written with standard markdown math delimiters:

- inline math uses `$...$`
- display math uses `$$...$$`

You still need a markdown renderer with math support enabled.

## Goal

For each decoder layer $\ell$, learn a projector

$$
W^\ell \in \mathbb{R}^{r \times d}, \qquad r \ll d,
$$

that maps each high-dimensional KV token key into a low-dimensional semantic space.
The model still attends to high-dimensional K/V vectors. The low-dimensional
representation is used only to decide which KV tokens to keep.

## Online Cache State

For each sequence and each layer, maintain a live cache state

$$
\mathcal{C}_t^\ell = \{(K_i^\ell, V_i^\ell, x_i^\ell, w_i^\ell)\}_{i=1}^{N_t},
$$

where:

- $K_i^\ell, V_i^\ell$ are the retained high-dimensional key/value vectors.
- $x_i^\ell = W^\ell \bar{k}_i^\ell$ is the projected feature of token $i$.
- $\bar{k}_i^\ell$ is the mean over KV heads for token $i$.
- $w_i^\ell$ is the cumulative attention importance of retained token $i$.

At token step $t$, the new token is appended to the live cache before attention is
computed, so the current query can still attend to itself.

## Attention Importance

At layer $\ell$, let the current query be $q_t^\ell$. Let the retained live keys and
values before compression be

$$
K_{1:N_t}^\ell, \qquad V_{1:N_t}^\ell.
$$

Attention is computed only over the retained live cache:

$$
\alpha_{t,i}^\ell = \mathrm{softmax}_i
\left(
\frac{\langle q_t^\ell, K_i^\ell \rangle}{\sqrt{d_h}}
\right).
$$

The per-token cumulative importance is updated online:

$$
w_i^\ell \leftarrow w_i^\ell + \sum_h \alpha_{t,i,h}^\ell.
$$

This replaces the old proxy $\lVert k_i \rVert$ used by the scaffold.

## Compression Trigger

After token $t$, define the cache budget

$$
B_t = \min\left(\lfloor \rho t \rfloor + s + r_w,\; t \right),
$$

where:

- $\rho$ is `semantic_kv_budget_ratio`
- $s$ is the sink window size
- $r_w$ is the recent window size

Compression is triggered whenever the live cache size exceeds $B_t$.

Sink tokens and recent tokens are always protected. Only middle tokens are candidates
for semantic selection.

## Greedy Semantic Selection

For each candidate token $i$, compute its projected feature

$$
x_i^\ell = W^\ell \bar{k}_i^\ell.
$$

Let $R$ be the current retained set of protected tokens plus previously selected middle
tokens. The greedy score is

$$
u_i = \log(w_i + \varepsilon) +
\log\left(\min_{j \in R} \lVert x_i - x_j \rVert_2 + \varepsilon\right).
$$

This is equivalent to maximizing

$$
w_i \cdot \min_{j \in R} \lVert x_i - x_j \rVert_2
$$

in hard inference, but is numerically safer for the soft relaxation.

Selection proceeds greedily without replacement until the middle-token budget is
filled.

## Straight-Through Relaxation

During training, the forward pass uses the hard greedy choice so the model still
attends to a hard subset of high-dimensional K/V tokens.

At greedy step $m$, for the remaining candidates:

$$
p_i^{(m)} = \operatorname{softmax}(u_i^{(m)} / \tau),
$$

$$
h_i^{(m)} = \operatorname{onehot}\left(\arg\max_i u_i^{(m)}\right),
$$

$$
z_i^{(m)} = h_i^{(m)} + p_i^{(m)} - \operatorname{stopgrad}(p_i^{(m)}).
$$

The selected high-dimensional slot is formed as

$$
\tilde{K}_m = \sum_i z_i^{(m)} K_i,
\qquad
\tilde{V}_m = \sum_i z_i^{(m)} V_i,
\qquad
\tilde{x}_m = \sum_i z_i^{(m)} x_i.
$$

In the forward pass, $z = h$, so the retained K/V is exactly the hard greedy choice.
In the backward pass, gradients flow through $p$, so the normal GRPO actor loss
updates $W$.

## Training Objective

The implemented objective is

$$
\mathcal{L}(W) =
\mathcal{L}_{\mathrm{GRPO}}(\text{compressed forward}; W)
+
\lambda \mathcal{L}_{\mathrm{cluster}}(W).
$$

There is no separate REINFORCE loss for selector actions in this implementation.
Instead, $W$ receives reward-conditioned gradients through the compressed forward path
itself.

### Cluster Regularizer

At each compression event, let the pre-compression projected tokens be

$$
X = \{x_i\}_{i=1}^{N},
$$

and let the retained centers after compression be

$$
C = \{c_j\}_{j=1}^{M}.
$$

The regularizer uses soft assignments

$$
a_{ij} = \operatorname{softmax}_j\left(-\lVert x_i - c_j \rVert_2^2 / T\right),
$$

importance-normalized within-cluster spread

$$
\operatorname{Within} =
\sum_i \hat{w}_i \sum_j a_{ij}\lVert x_i - c_j \rVert_2^2,
$$

and mean pairwise center separation

$$
\operatorname{Between} =
\frac{1}{|P|}\sum_{(j,k)\in P}\lVert c_j - c_k \rVert_2^2.
$$

The loss is

$$
\mathcal{L}_{\mathrm{cluster}} =
\frac{\operatorname{Within}}{\operatorname{Between} + \varepsilon}.
$$

## Training-Time Attention Path

In AReaL, each attention layer now:

1. Computes Q/K/V for the packed sequence.
2. Applies RoPE.
3. Runs an explicit online compressed-attention loop per sequence.
4. Updates cumulative attention importance from actual attention probabilities.
5. Compresses the live cache when the semantic budget is exceeded.
6. Stores compression events for the cluster regularizer.

This means the GRPO actor loss now depends on $W$, unlike the old scaffold which only
recorded projector statistics and then called the original attention unchanged.

## Rollout Path

In SGLang, semantic-KV rollout is implemented only for the `torch_native` attention
backend. The backend maintains per-request semantic state:

- retained KV pool indices
- cumulative importance scores
- logical sequence length

Prefill and decode both use the same hard greedy compression policy as training
inference mode. The rollout path therefore matches the hard forward semantics of the
training path.

## Serving-Side Compaction

The rollout path now also performs true allocator-side reclamation under the current
SGLang constraints:

- semantic state is keyed by a stable request id instead of `req_pool_idx`
- each layer reports its retained physical KV slots to a shared compaction manager
- the manager keeps per-request refcounts over token slots and frees a slot only after
  all semantic-KV layers have dropped it
- unfinished requests carry a placeholder `prefix_indices` tensor whose length is still
  the logical prefix length, while the real retained slot set lives in the semantic
  state

This makes serving-side memory reclamation correct for the current semantic-KV rollout
mode, but it requires the following restrictions:

- `torch_native` attention backend
- `page_size = 1`
- radix/prefix reuse disabled
- overlap scheduling disabled
- no speculative decoding or disaggregation

Under those constraints, rollout and training both attend only to the retained
high-dimensional K/V tokens and the allocator can reuse dropped KV slots immediately.
