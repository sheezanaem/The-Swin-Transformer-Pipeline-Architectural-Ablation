# Swin-Tiny Fine-Tuning with Architectural Ablation

**Course:** Advanced Computer Vision (PhD) — FAST School of Computing, FAST-NUCES Islamabad
**Assignment:** Assignment 1 — The Swin Pipeline & Architectural Ablation
**Dataset:** EuroSAT (RGB), 10 classes, 27,000 images
**Backbone:** `swin_tiny_patch4_window7_224` (pretrained on ImageNet-1k)

## 1. Overview

This repository implements an end-to-end pipeline for fine-tuning **Swin-Tiny**
on EuroSAT, followed by a controlled **architectural ablation** that injects two
modifications into the backbone:

| ID | Modification | Source |
|----|--------------|--------|
| **M1** | Replace shifted-window attention with **global attention** in every other transformer block | Article-critique improvement (mandatory) |
| **M2** | Replace **GELU** with **SiLU** activation throughout the backbone | Selection from the provided list |

Both models are trained with **identical hyperparameters and data splits**,
enabling a fair comparison. A full analysis of training curves, gradient norms,
and hypothesis outcome is documented in the accompanying IEEE report.

## 2. Results Summary

| Metric | Baseline | Ablation | Δ |
|---|---|---|---|
| Best Val Accuracy (top-1) | **0.9893** | 0.9869 | −0.24% |
| Last Val Accuracy | 0.9889 | 0.9854 | −0.35% |
| Last Val Loss | 0.0550 | 0.0565 | +0.0015 |
| Max raw L2 gradient norm | 23.90 | **45.42** | +90% |
| Mean raw L2 gradient norm | 8.69 | 15.71 | +81% |
| Mean clipped gradient norm | 0.882 | 0.972 | +10% |
| Training time (30 epochs) | 78m 55s | 78m 56s | +1 s |

**Hypothesis outcome:** REJECTED. Global attention + SiLU did not improve accuracy
on EuroSAT (Δ = −0.24%), but the ablation produced a systematic increase in
gradient magnitude (~1.8×), which we analyze as a side effect of replacing
window-local attention with full-token attention.

Neither run approached the 10³ instability threshold; gradient clipping at 1.0
was consistently active but functioned as a safety margin rather than a
rescue mechanism.

## 3. Repository Structure

```
.
├── README.md                 # this file
├── requirements.txt          # Python dependencies
├── ACV_Assignment_1.ipynb    # full pipeline (Colab-ready)
└── acv_assignment_1.py       # full pipeline (script-ready)
```

## 4. Setup

### 4.1 Hardware

- **Recommended:** NVIDIA GPU with ≥ 12 GB VRAM (T4, P100, V100, A100, RTX 30/40-series)
- **Reproduced on:** Google Colab, Tesla T4 (15.6 GB), CUDA 12.x
- **CPU fallback:** supported but ~30× slower (≈ 40 h total)

### 4.2 Environment

Requires **Python 3.8+**, **PyTorch ≥ 2.0**, **timm ≥ 0.9**.

```bash
# Clone
git clone https://github.com/<your-username>/swin-eurosat-ablation.git
cd swin-eurosat-ablation

# Create env (conda recommended)
conda create -n swin_ablation python=3.10 -y
conda activate swin_ablation

# Install
pip install -r requirements.txt
```

If you use a GPU, install the CUDA-enabled PyTorch build first
(see https://pytorch.org/get-started/locally/):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 5. Reproducing the Results

### 5.1 Full pipeline (baseline + ablation)

```bash
python acv_assignment_1.py
```

This will:

1. Download EuroSAT (≈ 270 MB) into `./data/` on first run.
2. Perform a **stratified 80/20 split** with `seed=42`.
3. Build `BaselineSwinT` with model-init `seed=123` and train for 30 epochs.
4. Build `AblationSwinT` with the same seed and train under identical settings.
5. Save all checkpoints, metrics `.pkl` files, and figures to `./logs/` and `./plots/`.

### 5.2 Notebook (Colab)

Open `ACV_Assignment_1.ipynb` and run cells sequentially.
On a T4, expect ≈ 80 minutes per model (≈ 160 minutes total).

### 5.3 Reproducibility guarantees

| Setting | Value |
|---|---|
| Data split seed | **42** |
| Model init seed | **123** |
| Determinism | `torch.backends.cudnn.deterministic = True`, `benchmark = False` |
| AMP | fp16 (`torch.amp.autocast`) |

Minor non-determinism can still arise from cuDNN kernels; run-to-run
accuracy variance is typically < 0.1%.

## 6. Fixed Hyperparameters (identical for both runs)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 5 × 10⁻⁵ |
| Weight Decay | 1 × 10⁻⁴ |
| Batch Size | 32 (reduced only if OOM — see § 8) |
| Epochs | 30 |
| Scheduler | Cosine annealing with 5-epoch linear warmup (per-batch stepping) |
| Image Resolution | 224 × 224 |
| Augmentation | RandomResizedCrop(224), RandomHorizontalFlip, Normalize(ImageNet stats) |
| Gradient Clipping | 1.0 (global L2, applied every batch) |
| Random Seed (split) | 42 |
| Random Seed (model init) | 123 |

## 7. Architectural Modifications (Ablation)

### M1 — Global attention in every other block
- For each Swin stage, block indices `0, 2, 4, …` have their `WindowAttention`
  replaced with a `GlobalAttention` module.
- `GlobalAttention` performs scaled dot-product attention over **all** spatial tokens
  (`N = H × W`) instead of the standard 7×7 window.
- QKV, projection, and dropout shapes are preserved to match Swin's block contract.
- The pre-existing relative-position bias is unused in these blocks (consistent
  with removing window locality).

### M2 — GELU → SiLU
- Every `nn.GELU` module inside the backbone is recursively swapped for `nn.SiLU`.
- Head activations are **unchanged** (head is a single `nn.Linear`, as in the
  baseline, per the assignment spec).

### What was **not** modified
- Classification head: identical to baseline (`nn.Linear(768, num_classes)`).
- Stem convolution, patch embedding, and LayerNorms: unchanged.
- Optimizer, scheduler, LR, WD, batch size, epochs, augmentation, seeds: unchanged.

## 8. Handling Out-of-Memory (OOM)

If GPU memory is insufficient at batch size 32:

1. The pipeline reduces `CONFIG['batch_size']` by a power of two (32 → 16).
2. `CONFIG['oom_triggered']` is set to `True`.
3. **The same reduced batch size is used for both baseline and ablation.**
4. The change is recorded in `logs/*_results.pkl` and reported.

For our reproduced run, **no OOM occurred**; batch size was 32 for both models.

## 9. Logged Metrics

For each epoch, we log:

- Train loss
- Validation loss
- Validation top-1 accuracy
- **Raw (pre-clip) L2 gradient norm** — averaged over batches
- **Clipped (post-clip) L2 gradient norm** — averaged over batches
- Learning rate

For Oxford 102 Flowers, we additionally log **macro-F1** on the validation set.

All metrics are persisted to `logs/<run>_results.pkl` for offline plotting.

## 10. Figures Produced

| File | Contents |
|---|---|
| `plots/sample_train_batch.png` | 4 × 4 grid of augmented training images |
| `plots/comparison.png` | 2 × 3 overlay: train loss, val loss, val top-1 accuracy, raw grad norm (log-scale, 10³ line), clipped grad norm, LR schedule |
| `plots/macro_f1.png` | (Flowers102 only) macro-F1 bar chart |

## 11. Key Findings

1. **Accuracy:** The ablation underperforms the baseline by **0.24%** top-1
   (98.69% vs. 98.93%). The hypothesis is rejected — the modification does not
   help on EuroSAT.
2. **Gradient scale:** The ablation's raw L2 gradient norm is systematically
   **~1.8× larger** (mean 15.7 vs. 8.7). We attribute this to full-token softmax
   attention having higher entropy than window-local attention.
3. **Stability:** Neither run approaches the 10³ instability threshold. Gradient
   clipping is active in both but is a safety margin, not a rescue.
4. **Cost:** Despite global attention being theoretically much more expensive
   per modified block, wall-clock training time is unchanged (78m 56s vs. 78m 55s)
   because attention represents a minority of Swin-T's total compute at 224×224.

## 12. Limitations & Future Work

- Only one architectural variant (global-attention-every-other-block + SiLU) was tested; a sweep over `global_every` and separate ablation of M1/M2 would isolate each modification's effect.
- Results are reported from a single seeded run per model; repeated runs with multiple seeds would allow variance estimates around the reported deltas.
- The hypothesis was evaluated on EuroSAT only — a texture/locality-dominated remote-sensing dataset — so conclusions may not transfer to datasets with more global structure.

## 13. Citation

If you use this code, please cite the accompanying report:

```bibtex
@techreport{swin-ablation-2026,
  title       = {Swin-Tiny Fine-Tuning with Architectural Ablation on EuroSAT},
  author      = {<Sheeza Naeem>},
  year        = {2026},
  institution = {FAST School of Computing, FAST-NUCES Islamabad},
  note        = {Advanced Computer Vision (PhD), Assignment 1}
}
```

## 14. Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) (Hugging Face)
- [torchvision](https://pytorch.org/vision/stable/index.html) — EuroSAT dataset
- Swin Transformer: Liu et al., *"Swin Transformer: Hierarchical Vision Transformer
  using Shifted Windows"*, ICCV 2021.
