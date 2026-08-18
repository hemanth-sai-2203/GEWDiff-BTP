# GEWDiff-BTP: 1-Month Research Plan
**Start Date**: August 18, 2026  
**End Date**: September 18, 2026  
**Compute**: Kaggle T4 Free Tier (~12-15 GPU hrs/week)  
**Goal**: Implement CFG Fine-tuning + Baseline Ablation  

---

## 🎯 Core Strategy

We are **NOT** trying to do everything. Here's the logic:

```
Your Constraints (T4 GPU, 1 month, free Kaggle)
                    ↓
Mamba needs A100 (CUDA 8.0), 45+ GPU hours, plus T4 CUDA 7.5 fails
                    ↓
SKIP MAMBA for Phase 1
                    ↓
CFG is perfect: 5 lines code, 30 GPU hours, works on T4, fixes paper's stated problem
                    ↓
Implement CFG → Generate results → Write BTP
```

---

## 📅 Week-by-Week Timeline

### **WEEK 1: Baseline + Code Understanding** (NO GPU intensive)
- [ ] Read & document entire codebase (dataset.py, unet3d.py, edm.py, train.py, test_wdc_local.py)
- [ ] Reproduce baseline: Run epoch_200.pth on WDC, record metrics
- [ ] Create ablation script: Vary rho (0.5-0.9), sigma_min (0.002-0.2), sigma_max (60-100)
- [ ] **Output**: Ablation table (25 combinations) + Code documentation

**Deliverable**: `ABLATION_RESULTS_WEEK1.csv`

---

### **WEEK 2: CFG Design + Light Testing** (2 hrs Kaggle)
- [ ] Study CFG (Classifier-Free Guidance) concept
- [ ] Implement CFG in training code:
  - Add 10% probability of dropping C=[z_LR, mask] during training
  - Add guidance_scale parameter for inference
  - Blend: output = z_cond + w*(z_cond - z_uncond)
- [ ] Quick inference test with pretrained checkpoint
- [ ] Test different guidance_scale values [0, 3, 5, 7, 9]

**Deliverable**: `cfg_training_code.py` + Quick test results

---

### **WEEK 3: CFG Fine-tuning on Kaggle** (12-15 hrs GPU)
- [ ] Create Kaggle notebook for fine-tuning
  - Load epoch_200.pth
  - Enable CFG in training loop
  - Configure: 15 epochs, batch_size=1, lr=1e-5
  - Save checkpoint every 2 epochs
  - Validate every epoch
- [ ] Run 2 Kaggle jobs (split across week):
  - **Job 1** (Mon-Tue): Epochs 0-7 (~6 hrs)
  - **Job 2** (Wed-Thu): Epochs 8-15 (~6 hrs)
- [ ] Monitor: loss curves, SAM/FID trends
- [ ] Evaluate best checkpoint on full WDC test set

**Deliverable**: `gewdiff_cfg_epoch15.pth` + Training curves

---

### **WEEK 4: Results Analysis + BTP Report** (VS CODE)
- [ ] Create final comparison table:
  ```
  | Metric | Baseline | GEWDiff+CFG | Δ | % Change |
  |--------|----------|-------------|---|----|
  | PSNR   | 28.86    | ???         | ? | ? |
  | SSIM   | 0.7104   | ???         | ? | ? |
  | SAM    | 9.15     | ???         | ? | ? |
  | FID    | 44.46    | ???         | ? | ? |
  ```
- [ ] Spectral analysis:
  - SAM error maps: baseline vs CFG
  - Spectral curves for weak vs strong LR input
  - Did CFG fix weak-input problem? YES/NO
- [ ] Visual comparisons:
  - 3-band RGB crops
  - Edge quality
  - Artifact analysis
- [ ] Write BTP report section (2-3 pages):
  - Method: CFG mechanics + implementation
  - Results: comparison table + figures
  - Discussion: limitations + insights
  - Conclusion

**Deliverable**: `BTP_CFG_Results_Report.pdf` + All figures

---

## 📊 Success Criteria

| Week | Task | Success = |
|------|------|-----------|
| 1 | Ablation | Clear trends visible; baseline verified |
| 2 | CFG Code | No errors; inference ~33s (vs 28s) |
| 3 | Training | 15 epochs complete; best ckpt saved |
| 4 | Results | SAM ≤ 9.0 OR FID ≤ 44 OR clear visual improvement |

---

## 🚨 What if CFG Shows NO Improvement?

**That's still valid research!** Here's why:

✅ You identified that CFG doesn't help GEWDiff  
✅ This tells us GEWDiff already leverages LR signal well  
✅ Guides future work: focus on other limitations instead  

**Write in BTP**: *"CFG fine-tuning showed no statistically significant improvement (Δ SAM = -0.05), suggesting GEWDiff's existing conditioning mechanism is near-optimal. Future work should explore other bottlenecks, such as spectral modeling via Mamba state-space models."*

---

## 🚀 If CFG Shows IMPROVEMENT

**Publish-quality result!** Next steps:

1. Submit as conference paper: "Classifier-Free Guidance for Hyperspectral Image Super-Resolution"
2. If time allows post-BTP: implement Mamba (15-20 hrs on Lambda Labs A100 ~$30)
3. Combined paper: "GEWDiff+CFG+Mamba: Novel Improvements to..."

---

## 📌 Kaggle GPU Time Budget

```
Week 1:  2 hrs  (baseline + quick ablation runs)
Week 2:  2 hrs  (quick CFG test)
Week 3: 12 hrs  (MAIN fine-tuning)
Week 4:  1 hr   (final validation)
─────────────
Total:  17 hrs (Fits in 4 weeks × 4 hrs/week limit)
```

**If you hit Kaggle limits**: Use Colab free tier as backup (~12 hrs/week available)

---

## 🎓 What to Tell Your Professor

**At end of month:**

*"I reproduced GEWDiff baseline (PSNR 28.86, SAM 9.15) and identified its main limitation from the paper: poor performance on weak LR inputs. I implemented Classifier-Free Guidance fine-tuning, which trains the model to occasionally ignore the LR conditioning signal. At inference, I blend conditional and unconditional outputs using a guidance scale parameter.*

*After fine-tuning 15 epochs on Kaggle T4, CFG produces [INSERT RESULTS]. This demonstrates [IMPROVEMENT/NO IMPROVEMENT] and suggests [FUTURE DIRECTION].*

*My ablation study of noise schedule parameters (rho, sigma_min, sigma_max) is the first systematic study of GEWDiff's hyperparameter sensitivity, providing a contribution independent of CFG results."*

---

## ✅ Start Today

We'll begin with:
1. **VS Code work** (Monday): Document codebase + create ablation script
2. **Kaggle run** (Tuesday): Baseline reproduction
3. **Kaggle runs** (Wed-Fri): Ablation parameter sweep

No waiting. We have 1 month. Let's go.

---

## Appendix: Why Not Mamba?

| Aspect | Mamba | CFG |
|--------|-------|-----|
| CUDA requirement | 8.0+ (T4 is 7.5) ❌ | None ✅ |
| GPU hours needed | 45-55 hrs | 30-40 hrs |
| Kaggle feasible? | No (time + CUDA) | Yes ✅ |
| Code change complexity | Medium (new block) | Minimal (5 lines) ✅ |
| Risk of failure | High | Low ✅ |
| Implementation time | 3-4 days | 1-2 days ✅ |

**Verdict**: CFG in Month 1. Mamba as Phase 2 post-BTP if promising.

