# autoresearch/mar10 Optimization Analysis

50 commits on the branch, 76 experiment runs logged in `results.tsv`.
Hardware: 2x RTX 3060 12GB, Linux, PyTorch 2.9.1+cu128.

**Best result: val_bpb=0.5175** (DEPTH=3, n_embd=320, AR=96, 997 steps)
Starting baseline: val_bpb=1.228 (DEPTH=8, 50M params) — 58% improvement.

---

## Phase 1: Model Size (val_bpb 1.228 → 0.702)

| # | Experiment | Result | Status |
|---|-----------|--------|--------|
| 1 | Baseline DEPTH=8, 50M params | 1.228 | **baseline** |
| 2 | torch.compile on Linux | 1.227 | discard (no MFU gain) |
| 3 | DEPTH=6 (~26M params) | 1.029 | keep |
| 4 | DEPTH=4 (~11M params) | 0.794 | keep |
| 5 | WARMDOWN 0.5→0.3, DEPTH=4 | 0.775 | keep |
| 6 | WARMDOWN 0.3→0.2, DEPTH=3 | 0.757 | keep |
| 7 | WARMDOWN 0.2→0.1, DEPTH=2 | 0.707 | keep |
| 8 | DEPTH=2 batch=32 (autotune refresh) | 0.702 | keep |

**Key insight:** Fewer layers = more steps in 5min = dramatically lower loss. DEPTH=1 was too shallow (0.750).

## Phase 2: Width & LR Tuning at DEPTH=2 (0.702 → 0.651)

| # | Experiment | Result | Status |
|---|-----------|--------|--------|
| 9 | WINDOW_PATTERN=SL (global attn) | 0.702 | discard (no gain) |
| 10 | MATRIX_LR 0.04→0.06 | 0.697 | keep |
| 11 | ASPECT_RATIO=128, n_embd=256 | 0.678 | keep |
| 12 | MATRIX_LR 0.06→0.08 | 0.677 | keep |
| 13 | MATRIX_LR=0.10 | 0.677 | discard |
| 14 | EMBEDDING_LR=0.3 | 0.682 | discard (0.6 better) |
| 15 | ADAM beta1=0.9 | 0.685 | discard (0.8 better) |
| 16 | n_embd=384 wider | 0.738 | discard (too few steps) |
| 17 | WARMDOWN=0.15 | 0.673 | keep |
| 18 | WARMDOWN=0.2 | 0.677 | discard |
| 19 | WD=0.1 at WARMDOWN=0.15 | 0.675 | discard (marginal) |
| 20 | UNEMBEDDING_LR=0.04 | 0.684 | discard (low 0.004 acts as regularizer) |
| 21 | SCALAR_LR=0.1 | 0.678 | discard |
| 22 | WARMUP_RATIO=0.05 | 0.721 | discard (wastes steps) |
| 23 | FFN_EXPANSION=8 | 0.685 | discard (slower, no gain) |
| 24 | HEAD_DIM=64, AR=96, n_embd=192 | **0.651** | keep (129 steps, new best) |
| 25 | GQA/MQA (kv_heads=1) | 0.769 | discard (slow on RTX 3060) |
| 26 | EMBEDDING_LR=1.2 | 0.652 | discard |

**Key insight:** HEAD_DIM=64 with AR=96 was a big architecture win. GQA/MQA too slow for this GPU.

## Phase 3: Batch Size Revolution (0.651 → 0.522)

| # | Experiment | Result | Status |
|---|-----------|--------|--------|
| 27 | TOTAL_BATCH=2^17 | 0.602 | keep (482 steps) |
| 28 | TOTAL_BATCH=2^16 | 0.563 | keep (939 steps) |
| 29 | TOTAL_BATCH=2^15 | 0.563 | keep (1832 steps, half VRAM) |
| 30 | n_embd=256 (wider, 2^15 batch) | 0.551 | keep (1549 steps) |
| 31 | WARMDOWN=0.25 at 2^16 | 0.560 | keep |
| 32 | n_embd=256 WARMDOWN=0.25 | 0.546 | keep |
| 33 | n_embd=320 AR=160 | 0.537 | keep (1293 steps) |
| 34 | n_embd=384 AR=192 | **0.534** | keep (1136 steps) |
| 35 | n_embd=448 | 0.536 | discard (too wide) |
| 36 | MATRIX_LR=0.06 for 384 | 0.532 | keep |
| 37 | MATRIX_LR=0.05 | 0.531 | keep (marginal) |
| 38 | WARMDOWN=0.30 | 0.529 | keep |
| 39 | WARMDOWN=0.40 | 0.527 | keep |
| 40 | WARMDOWN=0.50 | 0.526 | keep |
| 41 | WARMDOWN=0.60 | 0.525 | keep |
| 42 | WARMDOWN=0.70 | 0.527 | discard |
| 43 | WARMDOWN=0.80 | 0.525 | keep (tied, plateau) |
| 44 | WEIGHT_DECAY=0.1 | 0.525 | keep (marginal) |
| 45 | WD=0.3 | 0.526 | discard |
| 46 | EMBEDDING_LR=0.8 | **0.524** | keep |
| 47 | EMBEDDING_LR=1.0 / 1.2 | 0.525/0.526 | discard |
| 48 | FINAL_LR_FRAC=0.05 | 0.525 | discard |
| 49 | ADAM beta2=0.99 | **0.523** | keep |
| 50 | beta2=0.999 / 0.98 | 0.525/0.523 | discard |

**Key insight:** Smaller batch = more optimizer steps was the single biggest lever. Then long warmdown (60%) and beta2=0.99 squeezed out the rest.

## Phase 4: Depth Revisited (0.523 → 0.5175)

| # | Experiment | Result | Status |
|---|-----------|--------|--------|
| 51 | DEPTH=3 n_embd=256 AR=80 | 0.523 | discard (marginal) |
| 52 | DEPTH=3 n_embd=320 AR=96 | **0.5175** | keep (997 steps, NEW BEST) |
| 53 | DEPTH=3 n_embd=384 AR=128 | 0.519 | discard (fewer steps) |
| 54 | DEPTH=4 n_embd=256 AR=64 | 0.518 | discard (tied) |
| 55 | MATRIX_LR 0.04/0.06 for DEPTH=3 | 0.518/0.517 | discard (0.05 fine) |
| 56 | WARMDOWN 0.50/0.70 for DEPTH=3 | 0.519/0.518 | discard (0.60 optimal) |

**Key insight:** DEPTH=3 with 320 dim edges out DEPTH=2 with 384 dim. The extra layer's representational capacity is worth the step count penalty.

---

## Hyperparameter Search Summary

What's been fully explored and the optimal value found:

| Hyperparameter | Values tested | Optimal |
|---|---|---|
| Depth | 1, 2, 3, 4, 6, 8 | 3 |
| Width (n_embd) | 128, 192, 256, 320, 384, 448 | 320 (for DEPTH=3) |
| TOTAL_BATCH_SIZE | 2^15, 2^16, 2^17, 2^18 | 2^15 |
| MATRIX_LR | 0.04, 0.05, 0.06, 0.08, 0.10 | 0.05 |
| WARMDOWN_RATIO | 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80 | 0.60 |
| EMBEDDING_LR | 0.3, 0.4, 0.6, 0.8, 1.0, 1.2 | 0.8 |
| WEIGHT_DECAY | 0.0, 0.1, 0.2, 0.3 | 0.1 |
| ADAM beta2 | 0.95, 0.98, 0.99, 0.999 | 0.99 |
| ADAM beta1 | 0.8, 0.9 | 0.8 |
| HEAD_DIM | 64 vs default | 64 |
| FFN_EXPANSION | 4, 8 | 4 |
| UNEMBEDDING_LR | 0.004, 0.04 | 0.004 |
| SCALAR_LR | 0.1, 0.5 | 0.5 |
| WARMUP_RATIO | 0.0, 0.05 | 0.0 |
| FINAL_LR_FRAC | 0.0, 0.05 | 0.0 |
| WINDOW_PATTERN | SS, SL | no difference |
| GQA/MQA | MHA, MQA (kv_heads=1) | MHA (MQA too slow on 3060) |
| torch.compile | on, off | off (no benefit) |
| ASPECT_RATIO | 64, 80, 96, 128, 160, 192, 224 | 96 (for DEPTH=3) |

## Current Best Configuration

```
DEPTH = 3
ASPECT_RATIO = 96          # n_embd = 320 (5 heads x 64)
HEAD_DIM = 64
FFN_EXPANSION = 4
TOTAL_BATCH_SIZE = 2^15    # 32768
MATRIX_LR = 0.05
EMBEDDING_LR = 0.8
UNEMBEDDING_LR = 0.004
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.8, 0.99)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.60
FINAL_LR_FRAC = 0.0
USE_VE = True (alternating)
WINDOW_PATTERN = "SSSL"
```

Result: val_bpb=0.5175, 997 steps, ~4.4GB VRAM.

---

## Untested Experiments (ranked by potential)

### High confidence (likely to help)

1. **TOTAL_BATCH=2^14 (16384)** — Every batch halving has been a huge win (2^18→2^17→2^16→2^15). This is the most obvious gap in the search. Would roughly double step count to ~2000.

2. **SwiGLU/GeGLU activation** — The MLP uses `F.relu(x).square()` (squared ReLU). Gated activations (SwiGLU: `silu(gate) * up`) are the modern standard and typically outperform, especially at small scale. Requires splitting c_fc into gate+up projections. Adjust hidden dim to keep param count similar (e.g., `hidden = 8/3 * n_embd` instead of `4 * n_embd`).

3. **Muon ns_steps reduction (5→3 or 4)** — Each Newton-Schulz iteration in the Muon optimizer costs wall-clock time. Fewer iterations = faster steps = more steps in 5 minutes. The quality/speed tradeoff is worth testing.

### Medium confidence

4. **Weight tying** (share wte and lm_head) — Reduces total parameter count, freeing capacity for transformer layers. Standard practice in small language models.

5. **Softcap tuning** — Logit softcapping is hardcoded at 15. Never swept. Values like 20, 30, or removing it entirely could help.

6. **Value embedding coverage** — Currently alternating layers with gating. With DEPTH=3 that's 1-2 VE layers. Test `--no-ve` (faster forward pass = more steps) vs VE on every layer (more capacity).

7. **Cosine LR schedule** — Only linear warmdown has been tested. Cosine annealing sometimes gives a small edge, especially with the long cooldown phase already working well.

8. **DEPTH=3 with smaller width + smaller batch** — AR=80 (n_embd=256) at DEPTH=3 gave 0.523 with 2^15 batch. Combined with 2^14 batch (more steps), smaller width might close the gap.

### Lower confidence (exploratory)

9. **x0_lambda initialization** — Currently 0.1. This controls the skip-connection strength from the initial embedding. At DEPTH=3, a stronger residual (0.2, 0.3) might help since few layers must refine representations.

10. **Muon momentum schedule** — Currently ramps from 0.85→0.95 over 300 steps. Never tuned for the current config.

11. **Gradient clipping** — Not implemented. Could stabilize training and allow slightly higher learning rates.

12. **Window pattern for DEPTH=3** — With 3 layers, "SSSL" maps to "SSL" (layers 0,1=short window, layer 2=long). Try "LLL" (all global attention) — cost is minimal but information flow may improve.
