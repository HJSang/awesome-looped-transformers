# From first principles: how a looped architecture changes speculative decoding

> **Status:** research notes, 2026-09-05. Third note in the speculative-decoding series: the [survey and proposals](speculative-decoding-for-looped-transformers.md) list methods, the [DFlash/MTP note](spec-decoding-on-top-of-dflash-mtp.md) gives an engineering ladder, and this note derives *why* the loop matters from the definitions alone. No method names until §4.

---

## 1. What speculative decoding actually exploits

Three facts, none of which mention drafters.

1. **Decoding is bandwidth-bound.** At batch size 1, one forward pass costs roughly the time to stream the weights (plus KV) through the memory hierarchy; arithmetic is nearly free. A pass over $\gamma{+}1$ positions costs about the same as a pass over 1. Speculative decoding (SD) is an arbitrage on this fact: turn serial passes into one parallel pass over a *guess*.
2. **Token difficulty is heavy-tailed.** Most next tokens are nearly determined by their context; a few are not. Any cheap process that gets the easy ones right lets the expensive process settle several tokens per pass.
3. **The guess must be close to the target and cheap.** With $N$ the expected accepted run per cycle,
   $$\text{speedup}\;\approx\;\frac{N\cdot T_{\text{target}}}{T_{\text{draft}}+T_{\text{verify}}}.$$
   The tension: whatever approximates the target well tends to cost like the target. Every SD family is an answer to "proximity without paying for it": a smaller model (proximity via shared training data), heads or feature-level drafters (proximity via *reusing the target's own representation* and extrapolating one step), block drafters (same reuse, more positions per pass).

What the dense-model drafter fundamentally is: a **different function** from the target. Its error is a modelling error with no structure that can be measured before verification.

---

## 2. What a loop fundamentally is

A looped model is an iterated map on the residual stream:

$$s_0=\text{emb}(x),\qquad s_{k+1}=R(s_k;\ \text{context}),\qquad p^{(k)}=\text{readout}(s_k),\qquad \text{target}=p^{(K)}.$$

Four consequences follow from the definition alone.

**Cost.** Per token, the same weights are streamed $K$ times. Ouro-1.4B at $K=4$ has the decode latency of a 5.6B dense model and the memory of a 1.4B one. Loops multiply exactly the quantity SD reduces, so a looped model needs SD *more* than a dense one; and because each loop is itself bandwidth-bound, batching positions inside a loop is free.

**The natural sub-model is a truncation, not an approximation.** $p^{(k)}$ for $k<K$ is the same function stopped early. The draft–target gap is the truncation error of an iteration, $\|s_K-s_k\|$, not a mismatch between two networks.

**The trajectory is observable.** $s_1,\dots,s_K$ are computed anyway. If the map is near-contractive with rate $\rho$ (what stability training such as Parcae / residual scaling pushes toward), then
$$\|s_K-s_k\|\;\lesssim\;\frac{\rho^{k}}{1-\rho}\,\|s_1-s_0\|,$$
and the readout is Lipschitz in $s$, so the residual $\|s_{k+1}-s_k\|$ bounds how much $p^{(k)}$ can still move. The draft carries its own error estimate.

**The dependency graph is two-dimensional.** Loop $k$ at position $t$ needs

- (i) $s_{k-1}(t)$,
- (ii) the loop-$k$ KV of positions $<t$,
- (iii) the *identity* of token $t$, which needs the loop-$K$ readout of position $t{-}1$.

Only (iii) is a true serial dependency across positions. A dense model has the same dependency, but it is the *only* structure; here it sits inside a grid of (position, depth) cells, most of which do not depend on it.

```
depth ↑
  K   │ ■ ■ ■ ■ □ □ □ □         ■ = computed
  ⋮   │ ■ ■ ■ ■ ■ □ □ □         □ = not yet
  2   │ ■ ■ ■ ■ ■ ■ □ □
  1   │ ■ ■ ■ ■ ■ ■ ■ □   ← position t+1 can start loop 1 once a *guess*
  0   │ ■ ■ ■ ■ ■ ■ ■ ■     of token t exists (available after loop k of t)
      └────────────────→ position
```

---

## 3. Derived consequences for speculation

**C1. Draft error is a convergence residual, so acceptance is predictable before verification.**
In dense SD, draft failures are discovered only at verification, or by an extra confidence head. In a looped model the per-token residual, or an exit hazard trained on the same signal (Ouro's Stage-II gate), says in advance which tokens have converged. Draft depth can be chosen per token by a rule with a justification, not a heuristic: keep looping while the residual says the readout is still moving.

**C2. Draft compute is recycled, so accepted tokens cost exactly $K$ loops.**
Verification of a drafted position resumes from its draft state and runs loops $k{+}1..K$; the drafter's work is the verifier's first $k$ loops. In dense SD the drafter's work is discarded; even layer-skipping self-speculation must re-run the skipped layers. Overhead therefore comes only from *rejected* positions. With serial drafting the cycle is $\gamma k+(K-k)$ loop-steps for $\gamma$ tokens, so the ceiling as $\gamma\to\infty$ is
$$\text{speedup}_{\max}=K/k .$$
What remains binding is the serial chain of drafting one token after the next.

**C3. The serial chain can be pipelined across depth, with token identity as the only thing speculated.**
From the 2-D graph: position $t$ can begin loop 1 as soon as a *guess* of token $t{-}1$ exists, which is available after loop $k$ of position $t{-}1$, not loop $K$. The loop-$k$ KV of earlier positions is available because earlier positions are deeper in the pipeline. So a diagonal wavefront of about $K/k$ positions can be in flight, every cell computed once, total FLOPs per token unchanged at $K$ loops, and latency divided by up to $K/k$ when guesses are right. When a guess is wrong, everything to its right is invalidated: the ordinary SD rejection. This is the cleanest statement of what the loop gives:

> In a dense model, draft and target are two computations, one of which is thrown away. In a looped model they can be two *roles of the same computation*, and speculation degenerates to scheduling a (position × depth) grid under a single serial constraint.

The diffusion-forcing sampler of Geiping et al. ([2510.14961](https://arxiv.org/abs/2510.14961)) is this pipeline without the invalidation step, which is why it is fast (up to 5× on Huginn) and lossy. Adding invalidation, plus the rejection-sampling correction between $p^{(k)}$ and $p^{(K)}$ at the same position, makes it exact.

**C4. Speculation and adaptive depth harvest the same non-uniformity.**
The heavy tail in token difficulty is what makes drafts accepted, and it is what makes early exit save compute. A dense model can exploit it only by batching (SD). A looped model can exploit it along two axes at once: easy tokens are the ones that converge in few loops *and* the ones the pipeline guesses correctly. An exit gate trained on "did another loop reduce the loss" is, up to calibration, an acceptance predictor.

**C5. Verification can be depth-adaptive when the target is.**
If the deployed target is "loop until the exit rule fires" rather than "loop $K$ times", verification of a block stops per position at its exit loop and stays exact with respect to that target. This requires the exit rule to be a deterministic function of the state, which holds for a threshold rule on the gate CDF.

**C6. Drafter quality is a property of the target's training, not of a second model.**
A dense drafter must be trained to approximate the target. Here $p^{(k)}$ is good exactly to the extent that pretraining made shallow loops good language models: Ouro's per-loop loss and Huginn's random-$K$ training both do this implicitly; a depth-consistency objective (from any $s_k$ predict $p^{(K)}$) would do it explicitly. Improving the drafter and improving early-exit behaviour are the same fine-tune.

**C7. Memory stays bounded.**
The pipeline needs loop-$k$ KV only for in-flight positions. Ouro's finding that settled positions need only the last loop's KV for decoding (GSM8K 78.85 vs 78.92 with 4× less cache) means speculative KV scales with wavefront width, not with $K\times$ context.

---

## 4. What the loop does *not* give

- **Not a cheap independent drafter.** One loop of Ouro is a quarter of the target; a 10× smaller draft model is a tenth. With serial token-by-token drafting, a good small drafter beats naive depth-speculation (the [cost model](speculative-decoding-for-looped-transformers.md#40-cost-model-shared-by-all-proposals) caps it near 1.2–1.8× at $K=4$). The loop's advantage is reuse (C2) and pipelining (C3), not a lower draft price.
- **Shallow readouts may be weak.** Ouro-1.4B's loop-1 model is far below its loop-4 model (MMLU 41.2 vs 67.5), so guessing at loop 1 has low acceptance; loop 2–3 is the realistic guess depth, which caps the pipeline width at $K/2$–$K/3$. Huginn's $K=32$ is a different regime: guesses at loop 4–8 leave a lot of width.
- **Hardware dependence.** Everything above assumes weights are re-streamed per loop. On hardware where a small shared core stays resident on chip, loops become cheap and the bandwidth pressure that makes SD worthwhile largely disappears; the 2-D pipeline still helps, but for compute utilisation rather than bandwidth.
- **Exactness needs two things.** Positions computed with a guessed left context must be invalidated when the guess fails, and the rejection correction must compare the guess distribution $p^{(k)}$ against the true $p^{(K)}$ *at the same position*. Omitting either turns the method into a different sampler.

---

## 5. Summary

Speculative decoding buys back the serial dependency between tokens. A looped transformer moves most per-token work *off* that dependency into a depth axis that can be pipelined. A dense model must invent a second, cheaper function to guess with; a looped model already contains a ladder of cheaper functions, each a prefix of the real computation, each with a measurable distance to the target, and each of whose work is kept when the guess is right.

| Property | Dense transformer | Looped transformer |
|---|---|---|
| Drafter | a different function | the target stopped early |
| Draft error | unstructured modelling error | truncation residual, measurable per token |
| Draft compute on acceptance | discarded | recycled (verification resumes) |
| Serial constraint | full pass of token $t{-}1$ | only the *identity* of token $t{-}1$ |
| Parallelism available | across drafted positions | across positions **and** depth (wavefront ≈ $K/k$) |
| Adaptive compute | none without extra training | early exit and speculation share one signal |
| Drafter training | separate model / heads | property of the target's own objective |

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning* (Huginn). [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
- Geiping, Yang, Su et al. *Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models.* [arXiv:2510.14961](https://arxiv.org/abs/2510.14961)
- Zhu et al. *Scaling Latent Reasoning via Looped Language Models* (Ouro). [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)
- Prairie et al. *Parcae: Scaling Laws for Stable Looped Language Models.* [arXiv:2604.12946](https://arxiv.org/abs/2604.12946)
- Pappone, Crisostomi, Rodolà. *Two-Scale Latent Dynamics for Recurrent-Depth Transformers.* [arXiv:2509.23314](https://arxiv.org/abs/2509.23314)
- Elhoushi et al. *LayerSkip.* [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)
