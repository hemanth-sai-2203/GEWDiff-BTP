# GEWDiff + CFG Implementation Guide (4-Week Roadmap)

## Overview
This guide walks you through implementing Classifier-Free Guidance (CFG) fine-tuning on GEWDiff across 4 weeks using Kaggle free T4 GPU (48-60 free hours/month).

**Success Metrics:**
- Week 1: Ablation study completed (25 parameter combinations)
- Week 2: CFG code validated (no runtime errors)
- Week 3: 15 epochs CFG fine-tuning on Kaggle
- Week 4: Final results table + BTP report section

---

## Week 1: Ablation Study (Parameter Sweep, No Code Changes)

**Goal:** Establish baseline performance with different noise schedules  
**GPU Time:** ~2 hours  
**Output:** `ablation_results.csv` with 25 rows

### Step 1.1: Prepare Data on Kaggle

1. Create new **Kaggle Dataset** with:
   - `/input/gewdiff-checkpoint/epoch_200.pth` (pretrained)
   - `/input/wdc-dataset/data/test_wdc/` (test images)

2. Create **Kaggle Notebook** (blank)

### Step 1.2: Run Ablation Script

Copy this into Kaggle notebook **Cell 1:**

```python
# Setup
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "einops"], check=False)

# Import ablation script
exec(open('/kaggle/input/gewdiff-code/ablation_testing.py').read())

# Run
tester = AblationTester(
    checkpoint_path='/kaggle/input/gewdiff-checkpoint/epoch_200.pth',
    device='cuda'
)

results_df = tester.test_parameters(
    data_dir='/kaggle/input/wdc-dataset/data/test_wdc',
    rho_values=[0.5, 0.6, 0.7, 0.8, 0.9],
    sigma_min_values=[0.002, 0.02, 0.05, 0.1, 0.2],
    sigma_max_values=[60, 80, 100],
)

tester.save_results('/kaggle/working/ablation_results.csv')
```

### Step 1.3: Analyze Results

```python
# Cell 2: Visualize trends
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/working/ablation_results.csv')

# Plot: PSNR vs rho
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, sigma_max in enumerate([60, 80, 100]):
    subset = df[df['sigma_max'] == sigma_max]
    axes[i].plot(subset['rho'], subset['psnr'], 'o-', label='PSNR')
    axes[i].set_title(f'PSNR vs Rho (sigma_max={sigma_max})')
    axes[i].set_xlabel('Rho')
    axes[i].set_ylabel('PSNR (dB)')
    axes[i].grid()

plt.tight_layout()
plt.savefig('/kaggle/working/ablation_trends.png', dpi=100)
plt.show()

print("\nBest 5 configurations:")
best = df.nsmallest(5, 'sam')[['rho', 'sigma_min', 'sigma_max', 'sam', 'fid']]
print(best)
```

### Step 1.4: Download Results

1. Go to Kaggle notebook → Output
2. Download `ablation_results.csv` and `ablation_trends.png`
3. Save to: `docs/week1_ablation_results/`

**Expected Result:**
- CSV with 25 rows (5 rho × 5 sigma_min × 1 sigma_max × 1 baseline)
- SAM range: 9.12-9.20 (minor variations)
- Best config: likely (rho=0.7-0.8, sigma_min=0.002-0.02)

---

## Week 2: Code Preparation (Local, No GPU)

**Goal:** Implement and validate CFG code  
**GPU Time:** ~1 hour (light testing only)  
**Output:** CFG training + inference code ready

### Step 2.1: Review CFG Mechanics

CFG works in **two phases:**

#### Training Phase:
```
Standard: z_HR_pred = model(z_LR, z_HR, mask, edge)
CFG:      With p=0.1, set z_LR=0, mask=0, edge=0
          Model learns: "What if LR is unknown?"
```

#### Inference Phase:
```
Standard: output = model(z_LR, mask=mask, edge=edge)
CFG:      z_cond = model(z_LR, mask, edge)         # With guidance
          z_uncond = model(zeros, None, None)      # Without guidance
          output = z_cond + w * (z_cond - z_uncond)
          where w = guidance_scale ∈ [0, 9]
```

### Step 2.2: Examine CFG Wrapper

**File:** `cfg_finetuning_wrapper.py`

```python
from cfg_finetuning_wrapper import CFGElucidatedDiffusion

# Wrap your existing model
model = ElucidatedDiffusion(...)
model.load_state_dict(torch.load('epoch_200.pth'))

cfg_model = CFGElucidatedDiffusion(model, p_drop=0.1)
cfg_model.train()

# Now training automatically:
# - 90% of steps: normal conditioning
# - 10% of steps: zero conditioning
for batch in dataloader:
    loss, loss1, loss2, loss3 = cfg_model(
        batch['img_lr_hf'],
        batch['img_hr_hf'],
        batch['mask'],
        batch['edge']
    )
```

### Step 2.3: Test CFG Inference

**File:** `cfg_inference_testing.py`

```python
from cfg_inference_testing import CFGInferenceTester

tester = CFGInferenceTester('epoch_200.pth', device='cpu')  # CPU for quick test

# Test guidance scales [0, 1, 3, 5, 7, 9]
results = tester.test_guidance_scales(
    guidance_scales=[0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
    num_steps=50
)

# Analyze
tester.analyze_guidance_effect(results)
```

**Expected Output:**
```
Testing guidance_scale=0.0
  Mean: 0.0234 | Std: 0.0145 | Range: 0.5234
...
Optimal guidance scale (min variance): 5.0
  Output std: 0.0087
```

### Step 2.4: Validate on Test Image

```python
# Load one WDC test image
import tifffile
from pathlib import Path

test_image = tifffile.imread(Path('data/test_wdc/test_0.tif'))
print(f"Shape: {test_image.shape}, dtype: {test_image.dtype}")

# Resize to 64×64 (PCA latent space)
from torchvision.transforms import Resize
lr_image = Resize((64, 64))(torch.from_numpy(test_image[:, :, :20]))  # First 20 PCA bands

# Test inference (CPU)
# output = model.sample_with_cfg(
#     conditioning=lr_image,
#     guidance_scale=3.0,
#     num_steps=50
# )
```

**Deliverable:** All 3 files (wrapper, inference test, evaluation) run without errors ✓

---

## Week 3: Fine-tuning on Kaggle T4 (Main GPU Work)

**Goal:** Fine-tune 15 epochs with CFG  
**GPU Time:** ~12 hours total  
**Output:** `gewdiff_cfg_best_epoch_X.pth`

### Step 3.1: Setup Kaggle Notebook

Create **new Kaggle Notebook**, copy from: `kaggle_cfg_finetuning_notebook.py`

Key structure:
```
Cell 1: Setup (pip install, imports)
Cell 2: Load checkpoint
Cell 3: Wrap with CFG
Cell 4: Setup optimizer
Cell 5: Load data
Cell 6: Training loop (15 epochs)
Cell 7: Save results
```

### Step 3.2: Configure Training Parameters

In notebook, modify **CFGTrainingConfig:**

```python
config.num_epochs = 15
config.train_batch_size = 1          # Small batch for T4 memory
config.learning_rate = 1e-5          # Low for fine-tuning
config.cfg_p_drop = 0.1              # 10% conditioning dropout
config.cfg_guidance_scale = 3.0      # For inference (test 3-7 later)
config.num_timesteps = 50            # Denoising steps
```

### Step 3.3: Run Training

**Expected Training Timeline:**

| Epoch | GPU Time | Est. Runtime | Notes |
|-------|----------|--------------|-------|
| 0-4 | 1.5 hrs | 1.5h total | Monitor loss curves |
| 5-9 | 1.5 hrs | 3.0h total | Should see SAM improvement |
| 10-14 | 1.5 hrs | 4.5h total | Fine-tune guidance scale |
| Val | 0.5 hrs | 5.0h total | Full inference on test set |

**Stop/Checkpoint Strategy:**
- Save after every 2 epochs
- If loss plateaus, can stop early (don't need all 15)
- Select best checkpoint by lowest **SAM** (spectral fidelity)

### Step 3.4: Monitor Training Curves

```python
# Cell 7 in notebook
import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv('/kaggle/working/cfg_finetuning_history.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(history['epoch'], history['train_loss'])
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')

axes[0, 1].plot(history['epoch'], history['val_psnr'])
axes[0, 1].set_title('Validation PSNR')

axes[1, 0].plot(history['epoch'], history['val_sam'])
axes[1, 0].set_title('Validation SAM (lower is better)')

axes[1, 1].plot(history['epoch'], history['val_fid'])
axes[1, 1].set_title('Validation FID (lower is better)')

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=100)
plt.show()
```

### Step 3.5: Download & Archive

From Kaggle notebook → Output:
1. `gewdiff_cfg_best_epoch_X.pth` → Save to `checkpoints/`
2. `cfg_finetuning_history.csv` → Save to `docs/week3/`
3. `training_curves.png` → Save to `docs/week3/`

---

## Week 4: Results Evaluation & Report Writing

**Goal:** Compare baseline vs CFG, write BTP section  
**GPU Time:** 0 (CPU only)  
**Output:** `comparison_table.csv`, `BTP_CFG_Results_Section.pdf`

### Step 4.1: Run Evaluation Script

```python
from cfg_results_evaluation import CFGResultsEvaluator

evaluator = CFGResultsEvaluator(output_dir='./cfg_results')

# Load metrics
baseline = evaluator.load_baseline_metrics(None)  # Uses paper values
cfg = evaluator.load_cfg_metrics('docs/week3/cfg_finetuning_history.csv')

# Create comparison
df = evaluator.create_comparison_table(baseline, cfg)
evaluator.save_comparison_table(df)

# Visualize
evaluator.create_visual_comparison(baseline, cfg)

# Report
evaluator.write_summary_report(baseline, cfg, df)
```

### Step 4.2: Interpret Results

**Scenarios & Actions:**

#### Scenario A: Improvement (SAM ↓ 0.2-0.5°, FID ↓ 1-3)
```
BTP Section 4.3 Result:
"Fine-tuning with CFG improved spectral fidelity (SAM: 9.15° → 8.95°, Δ=-0.20°)
and visual quality (FID: 44.46 → 43.50, Δ=-0.96). This validates CFG's ability
to handle ambiguous LR inputs by learning unconditional generation."
```

#### Scenario B: No Change (|Δ| < 0.1)
```
BTP Section 4.3 Result:
"CFG fine-tuning showed no statistically significant improvement (SAM: 9.15° → 9.12°).
This ablation result indicates GEWDiff's existing conditioning mechanism is
already near-optimal. Future improvements should explore architectural changes
(e.g., Mamba bottleneck, learned wavelet filters) rather than training modifications."
```

#### Scenario C: Degradation (SAM ↑ > 0.2°)
```
BTP Section 4.3 Result (Negative Result):
"CFG fine-tuning degraded spectral fidelity (SAM: 9.15° → 9.45°, Δ=+0.30°).
Over-dropping of conditioning signals may cause the model to ignore valuable
LR semantic information. Recommendation: reduce p_drop to 0.05 or use
soft-weighting instead of hard dropout for future work."
```

### Step 4.3: Create Publication-Ready Comparison Table

**Table Format:**

| Metric | Baseline | GEWDiff+CFG | Δ | % Change | Better? |
|--------|----------|-------------|---|----------|---------|
| PSNR (dB) | 28.8600 | 28.9200 | +0.0600 | +0.21% | ✓ |
| SSIM | 0.7104 | 0.7115 | +0.0011 | +0.15% | ✓ |
| SAM (°) | 9.1500 | 9.1200 | -0.0300 | -0.33% | ✓ |
| FID | 44.4600 | 44.2300 | -0.2300 | -0.52% | ✓ |
| RMSE | 0.0569 | 0.0568 | -0.0001 | -0.18% | ✓ |
| Inference (s) | 28.0 | 33.6 | +5.6 | +20.0% | ✗ |

**Save as:**
- `comparison_baseline_vs_cfg.csv` (Excel/spreadsheet)
- `comparison_baseline_vs_cfg.tex` (LaTeX table for paper)

### Step 4.4: Write BTP Report Section

**File:** Create `docs/BTP_CFG_Results_Section.md`

**Template Structure:**

```markdown
# Results: Classifier-Free Guidance Fine-tuning

## 4.1 Method

Classifier-Free Guidance (CFG) is a training technique that teaches the model
to generate samples both with and without conditioning signals. During training,
we randomly drop the LR guidance signal z_LR with probability p_drop=0.1,
forcing the model to learn unconditional denoising paths.

At inference, we blend:
$$\hat{z} = z_{cond} + w \cdot (z_{cond} - z_{uncond})$$

where $w$ is the guidance scale (typically 3.0-7.5).

Implementation:
- Fine-tuned from pretrained epoch_200.pth checkpoint
- 15 epochs with lr=1e-5 on Kaggle T4
- Batch size: 1 (memory constraints on T4)
- Total GPU time: 12 hours

## 4.2 Results

[INSERT COMPARISON TABLE HERE]

Key findings:
- SAM improved by 0.3° (if true) / unchanged (if ablation)
- FID improved by 0.2-1.0 (if true) / unchanged (if ablation)
- Inference time increased by ~20% due to double-pass denoising
- No architectural changes, fully compatible with existing pipeline

## 4.3 Analysis

### Spectral Fidelity (SAM)
[INSERT INTERPRETATION: positive if SAM decreased]

### Visual Quality (FID)
[INSERT INTERPRETATION: positive if FID decreased]

### Inference Cost
The 20% inference time increase (28s → 33.6s) is acceptable given:
- Still runs on single T4 GPU
- Batch processing can amortize cost
- Optional parameter (can use w=1.0 for standard inference)

## 4.4 Ablation: Effect of Guidance Scale

Testing guidance scales from 0 to 9:
- w=0.0: Purely unconditional (poor quality, high variance)
- w=1.0: Standard conditional (baseline)
- w=3.0: Recommended (balanced)
- w>7.0: Over-conditioning (artifacts, less natural)

Optimal w=3.0-5.0 for WDC dataset.

## 4.5 Conclusion

CFG provides a simple, low-cost method to improve GEWDiff's robustness on
ambiguous LR inputs. [Insert conclusion based on actual results.]
```

### Step 4.5: Generate Figures

```python
# Create spectral curves comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: SAM error map (if available)
# Plot 2: Spectral curves for weak LR input

axes[0].set_title('GEWDiff Baseline')
axes[1].set_title('GEWDiff+CFG')

plt.tight_layout()
plt.savefig('docs/cfg_results/spectral_comparison.png', dpi=300)
```

### Step 4.6: Final Deliverables

Create `docs/BTP_CFG_Results/` with:

```
├── comparison_baseline_vs_cfg.csv
├── comparison_baseline_vs_cfg.tex (for LaTeX paper)
├── comparison_visualization.png (bar charts)
├── training_curves.png (loss, PSNR, SAM, FID over epochs)
├── spectral_comparison.png (SAM error maps)
├── CFG_Results_Summary.txt (text report)
└── BTP_CFG_Results_Section.md (full BTP section)
```

---

## Implementation Checklist

### Week 1
- [ ] Kaggle dataset created with checkpoint + WDC data
- [ ] Ablation script runs without errors
- [ ] 25 parameter combinations tested
- [ ] `ablation_results.csv` saved
- [ ] Trends analyzed (plots generated)
- [ ] Results downloaded locally

### Week 2
- [ ] `cfg_finetuning_wrapper.py` reviewed
- [ ] `cfg_inference_testing.py` runs on CPU
- [ ] Guidance scale test completed (0-9)
- [ ] Test image inference works
- [ ] No runtime errors on any of 3 files
- [ ] Optional: Quick GPU test on 1 image

### Week 3
- [ ] Kaggle notebook created from template
- [ ] Training config validated
- [ ] 15 epochs completed (or stopped early if plateaued)
- [ ] Loss curves show decreasing trend
- [ ] Best checkpoint selected by SAM
- [ ] `gewdiff_cfg_best_epoch_X.pth` downloaded
- [ ] `cfg_finetuning_history.csv` saved

### Week 4
- [ ] `cfg_results_evaluation.py` runs successfully
- [ ] Comparison table generated
- [ ] Visualization created
- [ ] BTP report section written (2-3 pages)
- [ ] All figures saved (300 DPI for print)
- [ ] Final summary in `CFG_Results_Summary.txt`

---

## Troubleshooting

### "CUDA out of memory" on Week 3
**Solution:** Reduce batch_size from 1 to use gradient accumulation, or use fp16 precision

### "Validation SAM not improving"
**Solution:** This is valid ablation result! Write as negative result and conclude GEWDiff is already well-conditioned

### "Training loss NaN at epoch 5"
**Solution:** Reduce learning rate to 1e-6, or check gradient clipping is enabled

### "Checkpoint doesn't load on inference"
**Solution:** Verify checkpoint contains `unet_state_dict` key, not entire model. Use `model.load_state_dict(...)`

---

## File Summary

| File | Purpose | Week | Input | Output |
|------|---------|------|-------|--------|
| `ablation_testing.py` | Parameter sweep baseline | 1 | epoch_200.pth | ablation_results.csv |
| `cfg_finetuning_wrapper.py` | CFG training wrapper | 2-3 | base model | None (used in train) |
| `cfg_inference_testing.py` | Test guidance scales | 2 | epoch_200.pth | cfg_inference_results.json |
| `kaggle_cfg_finetuning_notebook.py` | Training on Kaggle | 3 | epoch_200.pth | gewdiff_cfg_best.pth |
| `cfg_results_evaluation.py` | Compare & report | 4 | best.pth + results.csv | comparison_table.csv |

---

## Expected Timeline

```
Week 1 (Mon-Fri):    Ablation study
  Mon-Tue: Setup Kaggle, run ablation (2 hrs GPU)
  Wed:     Analyze results
  Thu-Fri: Document findings, download outputs

Week 2 (Mon-Fri):    Code validation
  Mon-Wed: Review CFG mechanics, run inference tests
  Thu:     Quick GPU test on 1 image
  Fri:     All code ready, no errors

Week 3 (Mon-Fri):    Fine-tuning
  Mon-Tue: Epochs 0-7 (6 hrs GPU)
  Wed-Thu: Epochs 8-15 (6 hrs GPU)
  Fri:     Validation, select best checkpoint

Week 4 (Mon-Thu):    Results & report
  Mon-Tue: Run evaluation, create tables/figures
  Wed:     Write BTP section
  Thu:     Final review, submit
```

---

## Next Steps After CFG

If CFG shows **positive results**, consider Phase 2:
- [ ] Combine CFG + Mamba on larger GPU (Colab A100)
- [ ] Submit joint paper: "GEWDiff Improvements: CFG & Spectral Modeling"

If CFG shows **no improvement**, move to Phase 2 immediately:
- [ ] Implement Learned Wavelet Filters
- [ ] Or: Mamba bottleneck on A100

Both paths lead to publishable work! ✓

---

## Questions?

Refer to:
- GEWDiff paper: "Generative Modeling for Hyperspectral Image Super-resolution"
- CFG paper: "Classifier-Free Diffusion Guidance" (Ho & Salimans 2021)
- Config details: `src/GEWDiff/train.py` (TrainingConfig class)
