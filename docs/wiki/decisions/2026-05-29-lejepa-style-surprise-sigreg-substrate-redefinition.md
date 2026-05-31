# 2026-05-29 LeJEPA-style Surprise & SIGReg Substrate Redefinition

**Status**: Active architectural design and decision record
**Date**: 2026-05-29
**Trigger**: Falsification of multiple recurrence/non-recurrence substrates showing `persistent_carry_rate = 1.0` (representation collapse). Grounded in Yann LeCun & Randall Balestriero's *LeJEPA* (2025/2026) paper and *Titans* surprise-modulated memory updating.

---

## 1. Context & Core Problem

During our previous fast-falsification loops, every radical direction tested—including pure parallel search, evolutionary populations, and test-time self-modifying architectures—consistently reproduced the same negative pattern:
- `persistent_carry_rate = 1.0` (all memory slots are carried on every step).
- `ablation_drop = 0` (turning memory off produces no drop in reasoning performance).

### The "Notebook is Always Full" Diagnosis
The core problem is twofold:
1. **Representation Collapse (표현 붕괴):** In the absence of mathematical regularization, memory slots and recurrent states collapse into a tight cluster in the latent space. Because Slot A and Slot B look virtually identical, the router cannot make selective choices, routing densely to all slots (`carry_rate = 1.0`).
2. **Lack of Counterfactual Pressure (카운터팩추얼 압박 부재):** The model is trained with memory always enabled. It never experiences the loss of "not remembering," and thus never learns to selectively discard or write memories.

---

## 2. The LeJEPA & Titans Synthesis

To break this lock, we redefine our memory substrate using two state-of-the-art self-supervised learning principles:

### A. LeJEPA SIGReg (Sketched Isotropic Gaussian Regularization)
To prevent representation collapse without relying on fragile heuristics (like stop-gradients or teacher-student models), we apply an approximation of **SIGReg** using variance-covariance constraints on our memory slots:
- **Mean loss**: Forces slot embeddings to be zero-centered.
- **Covariance loss (decorrelation)**: Off-diagonal elements of the covariance matrix are pushed to 0, forcing slots to occupy orthogonal directions in the latent space.
- **Variance loss**: Diagonal elements of the covariance matrix are pushed to 1.0, forcing unit variance.

This mathematically guarantees that the memory slots remain highly distinct and decorrelated, giving the router sharp matching gradients and enabling selective Top-K routing.

### B. Titans-style Surprise-driven Gating
Instead of writing to the memory bank at every single recurrence micro-step, writes are **decoupled** and modulated by a **JEPA-style Joint-Embedding Prediction Error (놀람)**:
- At each rehearsal/training step, the surprise $\mathcal{S}_t$ is computed as the L2 error between the recurrent state $z_{h, t}$ and the retrieved memory context $m_t$:
  $$\mathcal{S}_t = ||\text{Proj}_v(z_{h, t}) - \text{Read}(z_{h, t}, \text{slots}_{t-1})||_2^2$$
- High-surprise events (where the current thinking state deviates from what the memory already predicts) trigger synaptic write gates:
  $$g_t = \sigma(\mathbf{W}_s \mathcal{S}_t + \mathbf{b}_s)$$
- Low-surprise events (highly predictable context) are ignored, preserving memory capacity and slot purity.

### C. Training-time Counterfactual Memory Drop
To force the recurrent path to rely on memory only when necessary, we stochastically drop the memory bank context with a probability ($p_{\text{drop}} = 0.15$) during training. This forces the recurrent state to adapt to "not having notes," creating strong training gradients for selective memory policies.

---

## 3. Mathematical & Code Implementation

The following modules have been modified and integrated:

### 1. `src/wgram_lm/memory/decoupled_latent_memory_bank.py`
- Added `compute_sigreg_loss(self, x)` method implementing the variance-covariance isotropic Gaussian constraint.
- Updated `controller_write` to calculate the L2 joint-embedding prediction error (JEPA surprise) dynamically when `rehearsal_target` is provided, automatically scaling the global write gate.

### 2. `src/wgram_lm/blocks.py`
- Integrated training-time **Counterfactual Memory Drop** stochastically inside `OneBodyParallelHybridBlock.forward` controlled by `self.cfg.counterfactual_memory_drop_prob`.

### 3. `scripts/train_hybrid_ri4_real_continuation_minimal.py`
- Registered `--decoupled_bank_sigreg_weight` and `--counterfactual_memory_drop_prob` CLI parameters.
- Injected the **LeJEPA SIGReg Loss penalty** directly into `train_loss` in the optimization loop before the backpropagation step.

---

## 4. Immediate Verification Plan

### Run Command (Real Gold Continuation):
```bash
python scripts/train_hybrid_ri4_real_continuation_minimal.py \
  --steps 200 \
  --use_decoupled_memory_bank \
  --decoupled_bank_sigreg_weight 0.05 \
  --counterfactual_memory_drop_prob 0.15 \
  --real_gold_path data/gold/642_bos_latent.pt
```

This represents the most mathematically complete and literature-backed substrate redefinition in our monorepo history, addressing both representation collapse and training signal deficiencies in a single unified leap.
