# GEWDiff Baseline — WDC

## Run
Date: 2026-08-10
Dataset: WDC
Checkpoint: epoch_200.pth

## Configuration
- Compack Bands: 121
- PCA Bands: 20
- Timesteps: 50
- Epochs: 200
- Batch Size: 1
- Mask: True
- Edge: True
- L1 Lambda: 0.8
- L2 Lambda: 0.1
- L3 Lambda: 0.1
- Sigma Min: 0.002
- Sigma Max: 80
- Sigma Data: 0.5
- Rho: 7

## Metrics

| Metric | Value |
|---|---:|
| MPSNR | 33.7827 |
| MSSIM | 0.6920 |
| SAM | 9.1510 |
| CrossCorrelation | 0.6275 |
| RMSE | 0.05695 |
| FID | 47.8211 |
| LV Pred | 0.005330 |
| LV True | 0.007440 |

## Output Files
- resultdiff_2026-08-10 20:23:25.npy
- resultdiff_rwa_pca.png
- hist_2.png

## Notes
- 50/50 diffusion sampling steps completed successfully.
- T4 x2 GPU used.
- No runtime failure.
- Matplotlib clipping warnings occurred during visualization; inference and metric computation completed.

