# Starting from DFlash / MTP: how to exploit the loop on top of traditional speculative decoding

> **Status:** research notes, 2026-09-05. Companion to [Speculative Decoding for Looped Transformers](speculative-decoding-for-looped-transformers.md) (survey and three proposals) and the [Ouro walkthrough](looped-transformer-taxonomy.md#type-1--model-loop-flat-whole-stack). This note takes the opposite starting point: assume a state-of-the-art conventional method (DFlash block-diffusion drafting, or DeepSeek-style multi-token prediction) is already ported to a looped model such as Ouro, and ask what the loop architecture adds on top. Ideas are ordered by cost, cheapest first.

Notation: $\tau$ = one pass of the shared 24-layer stack (one loop) over a batch of positions in the memory-bound decode regime; $\tau_d$ = one drafter pass; $h^{(t)}$ = Ouro's state after loop $t$; $\lambda_t$ = Ouro's exit-gate hazard at loop $t$; $N$ = expected accepted tokens per cycle (including the corrected token).

---

## Step 0 — The baseline: porting DFlash / MTP to Ouro as-is

| | DFlash on Ouro | MTP on Ouro |
|---|---|---|
| Drafter | small block-diffusion model ([DFlash, 2602.06036](https://arxiv.org/abs/2602.06036)) that emits a whole block in one pass, conditioned on target hidden states fused into its KV at every layer | DeepSeek-V3-style modules ([2412.19437](https://arxiv.org/abs/2412.19437)): one extra transformer block per future token, input = main-model state fused with the next token's embedding |
| Target feature used | $h^{(4)}$ (the loop-4 state) | $h^{(4)}_i$ fused with $\text{emb}(x_{i+1})$ |
| Verification | one 4-loop pass over the block | one 4-loop pass over the block |
| Cycle cost | $\tau_d + 4\tau$ | $D\,\tau_{\text{mtp}} + 4\tau$ |
| KV | keep only the last loop's KV for accepted tokens (Ouro Table "KV cache sharing": 78.85 vs 78.92 GSM8K); drop rejected positions | same |
| Expected result | about the same speedup DFlash gets on a dense model of equal effective depth | same as dense MTP |

Nothing here uses the loop. It is the control condition for everything below.

---

## Step 1 — Cheap, drafter-only: condition the drafter on the loop trajectory

**Observation.** DFlash and EAGLE-3 both found that *which* target features the drafter sees dominates acceptance; EAGLE-3's multi-layer fusion alone was worth 6–12 acceptance points ([2503.01840](https://arxiv.org/abs/2503.01840)). A dense model offers layers that compute different things. Ouro offers four states of the **same** layer at increasing depth, plus a trained difficulty signal.

**Proposal.** Replace the drafter's conditioning vector $h^{(4)}$ with

$$\big[\,h^{(4)},\; h^{(4)}-h^{(3)},\; h^{(3)}-h^{(2)},\; h^{(2)}-h^{(1)},\; \lambda_{1:4}\,\big]$$

- The difference vectors are the "latent thoughts" the token needed. A trajectory that converged by loop 2 marks an easy token; one still moving at loop 4 marks a hard one.
- The hazards $\lambda_t$ are what Stage II of Ouro's training taught the gate: *stop when another loop stops helping*. That is a per-token difficulty estimate the target was explicitly optimised to produce.
- Use the same signals to **size the block**: long draft blocks after confident tokens, short ones after tokens whose $\lambda$ stayed low (the Kangaroo "double early exit" idea, with a signal that exists for free).

**Cost.** Zero at inference (all four states are computed anyway); one drafter training run. **First experiment.** Acceptance length on Spec-Bench with and without trajectory features, on Ouro-1.4B. Hypothesis: the gain concentrates on math and code, where loop 3 → 4 still changes the argmax.

---

## Step 2 — Systems, no training: depth-resumable verification and verify-while-draft

In a dense model the verify pass is atomic. In Ouro it is four separable loops, which opens three tricks.

### 2a. Resume, don't recompute
If a block position already carries a loop-$k$ state (from Step 3 below, or from an MTP module designed to emit one), verification runs loops $k{+}1..4$ only. Layer-skipping self-speculation (LayerSkip, Kangaroo) never had this: their skipped layers still have to be executed at verification. Here the drafter's compute is literally the verifier's first loops.

### 2b. Provisional acceptance from a shallow loop
Today the next draft block cannot start until the verify pass finishes, because the accepted prefix length is unknown. Ouro exposes $p^{(2)}$ after two of the four loops:

1. run loops 1–2 of the verify pass over the block;
2. predict the acceptance boundary from $p^{(2)}$ (and the gate);
3. start drafting the next block from the predicted last-accepted position **while loops 3–4 finish**;
4. confirm with the exact rejection rule on $p^{(4)}$; if the predicted boundary was wrong, discard the draft and redo it.

This is the pipelining idea of PPSD ([2509.19368](https://arxiv.org/abs/2509.19368)) without needing an early-exit head: the provisional signal is a legitimate readout the model was trained on. Ouro-1.4B's loop-2 vs loop-4 gap (MMLU 60.4 vs 67.5) suggests the prediction is right on most tokens and wrong mainly where a rejection would have happened anyway.

```
dense-style :  [verify L1 L2 L3 L4][draft][verify L1 L2 L3 L4][draft] ...
loop-aware  :  [L1][L2][L3][L4][L1][L2][L3][L4] ...
                        └── draft next block ──┘   (overlapped, confirmed at L4)
```

### 2c. Depth-truncated verification (throughput)
Positions beyond the predicted boundary need not run loops 3–4 unless the exact boundary turns out to be further. In the single-stream, memory-bound regime this saves FLOPs rather than latency; in batched serving those FLOPs are real capacity.

**Cost.** Scheduler work only. Note that vLLM/SGLang's fixed execution graph is exactly what broke Ouro's RL rollouts (Ouro §Reinforcement Learning Attempts); a depth-resumable speculative scheduler is a systems contribution in its own right.

---

## Step 3 — Training: make the shared stack its own drafter

Here the loop stops being an optimisation and becomes the drafter.

### 3a. MTP as an extra loop, not an extra module
DeepSeek's MTP adds a new transformer block per future token. In Ouro, reuse the shared stack instead: fuse $h^{(4)}_i$ with $\text{emb}(x_{i+1})$ through a small projection and run the **existing** 24 layers once more to predict $x_{i+2}$; chain $D$ times as in DeepSeek.

- Zero new transformer parameters (one projection per head).
- Train the fusion so the MTP output is a valid **loop-1 state of position $i{+}1$**; then Step 2a applies and verifying that position needs only loops 2–4.
- Honest cost: this drafter is a full stack pass, so $D$ heads cost $D\tau$, versus DeepSeek's one-block module at roughly $\tau/24$. MTP-as-loop wins on parameters and (likely) acceptance, not on raw draft speed.

### 3b. Loop 1 as a DFlash-style block drafter
DFlash trains a small denoiser to fill a masked block in one pass. Ouro's stack is already an iterative refiner, and Geiping et al. argue recurrent-depth models behave as causal diffusion models ([2510.14961](https://arxiv.org/abs/2510.14961)). Fine-tune the shared stack with a light **masked-future-block objective** alongside the LM loss so that:

- **loop 1** over positions $i{+}1..i{+}\gamma$, attending to the real prefix, produces the draft block (the drafter);
- **loops 2–4** over the same positions, resumed from those states, are the verifier.

The cycle is one loop of drafting plus three of verifying, about $4\tau$, the same as generating a single token today, while emitting the whole accepted prefix. No separate drafter, no separate KV, and the draft states are the verifier's first loop. This is the highest-ceiling item and the only one that needs a real training run to believe.

---

## Step 4 — Optional, changes the target: exit-aware verification

Ouro officially supports early exit at threshold $q$ (first loop where $1-\prod_{j\le n}(1-\lambda_j)\ge q$). If the target is defined as "Ouro with exit threshold $q$", each block position may stop verification at its own exit loop and the procedure stays exact **with respect to that target**. The gain is modest because a block waits for its slowest position, but it composes with Steps 1–3. Flag clearly: this is a different target distribution than fixed 4-loop decoding.

---

## Cost picture

| Configuration | Cycle cost | Tokens per cycle | Comment |
|---|---|---|---|
| Ouro, plain decoding | $4\tau$ | 1 | baseline |
| DFlash on Ouro (Step 0) | $\tau_d + 4\tau$ | $N_{\text{DFlash}}$ | same as on a dense model; $\tau_d$ small |
| + trajectory features (Step 1) | same | $N \uparrow$ | drafter-only change |
| + verify-while-draft (Step 2b) | $\approx 4\tau$ | $N$ | hides $\tau_d$; matters more for MTP chains |
| MTP-as-loop, $D$ heads (Step 3a) | $D\tau + 3\tau$ | $\le D{+}1$ | zero extra params; drafter is a full loop |
| Loop-1 block drafter (Step 3b) | $\approx 4\tau$ | $N_{\text{loop1}}$ | drafter = verifier's first loop; needs fine-tune |
| Naive depth-speculation (for reference) | $\gamma(p{+}1) + 3 + p$ | $(1-\alpha^\gamma)/(1-\alpha)$ | capped near 1.2–1.8× at $K=4$; see companion note |

---

## Why start from DFlash rather than from "draft with fewer loops"

The drafter's cost fraction decides speculative decoding. A one-loop draft of Ouro costs a quarter of the target; DFlash's drafter costs a tenth or less. So DFlash-on-Ouro already beats naive depth-speculation before any loop-specific idea is added. What the loop then contributes is not a cheaper drafter but three things a dense model cannot offer:

1. **richer conditioning** — four depth-aligned states plus a trained difficulty gate (Step 1);
2. **a splittable verify pass** that can be resumed and overlapped (Step 2);
3. **with training, one set of weights that is both drafter and verifier**, distinguished only by loop count (Step 3).

## Recommended order

1. Steps 1 + 2 together on Ouro-1.4B with an off-the-shelf DFlash or EAGLE-3 drafter, measured against the Step-0 port (acceptance length, tokens/s, exactness vs 4-loop greedy).
2. Step 3a if parameter budget matters more than latency.
3. Step 3b as the paper, if Steps 1–2 show the loop states carry signal.

## References

- DFlash: Block Diffusion for Flash Speculative Decoding. [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)
- DeepSeek-V3 Technical Report (multi-token prediction). [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- EAGLE-3: Scaling up Inference Acceleration via Training-Time Test. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting. [arXiv:2404.18911](https://arxiv.org/abs/2404.18911)
- LayerSkip. [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)
- Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding (PPSD). [arXiv:2509.19368](https://arxiv.org/abs/2509.19368)
- Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models. [arXiv:2510.14961](https://arxiv.org/abs/2510.14961)
- Scaling Latent Reasoning via Looped Language Models (Ouro). [arXiv:2510.25741](https://arxiv.org/abs/2510.25741) · [HF config / modeling code](https://huggingface.co/ByteDance/Ouro-1.4B)
