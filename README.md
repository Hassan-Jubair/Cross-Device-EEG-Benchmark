# Signal-Space Spatial Harmonization and Streaming Covariance Alignment for Calibration-Free Cross-Dataset Motor Imagery BCIs

Official code, unit tests, and statistical reproduction repository for the manuscript:
> **Signal-Space Spatial Harmonization and Streaming Covariance Alignment for Calibration-Free Cross-Dataset Motor Imagery BCIs**  
> Hassan Jubair

---

## 1. Overview & Key Findings

This repository implements a unified signal-space harmonization and causal streaming covariance alignment framework evaluated across **8 public MI-EEG corpora ($N = 317$ subjects, $35,349$ binary left- vs. right-hand trials)**:
- **Primary 5-Dataset Native-Channel Core ($N = 254$ subjects, $29,558$ trials):** `PhysioNet`, `Cho2017`, `Lee2019-MI`, `Shin2017A`, and `Weibo2014`.
- **Extended 7-Dataset Healthy LODO Pool ($N = 267$ subjects, $33,349$ trials):** Adds `BNCI2014-001` and `Zhou2016`.
- **External Clinical Stress-Test Cohort ($N = 50$ acute stroke patients, $2,000$ trials):** `Liu2024` (*Scientific Data*, 2024).

### Summary of Main Empirical Results
1. **Dense-First 10–05 Spherical Spline Harmonization:** Reconstructing displaced sensorimotor electrodes (`C3`/`C4`) from native 10–05 neighbors prior to channel selection and Common Average Referencing (CAR) reduces spline NRMSE by **60%** (`0.257` vs. `0.635`), improves decoding on hybrid EEG-fNIRS caps (`Shin2017A`) by **+3.04 to +4.99 pp**, and prevents a **4.54 pp** subject-level (**4.40 pp** macro, 95% bootstrap CI `[+3.43, +5.38]` pp) LODO degradation under simulated 8-channel optode masking.
2. **Single-Trial Global-RMS Scaling vs. Channel Z-Scoring:** Preserves inter-channel ERD/ERS variance ratios on the covariance diagonal, outperforming channel-wise z-scoring by **+2.40 pp** across subjects (**+1.93 pp** macro, 95% CI `[+1.35, +2.52]` pp, cluster LME $p < 0.0001$) while matching offline multi-trial per-subject scaling ($\Delta = -0.06$ pp, TOST equivalent).
3. **Causal Streaming Covariance Alignment ($303.9\ \mu	ext{s/trial}$):** Trace-regularized recursive online EA ($\kappa = 0$) and stream-matched Ledoit–Wolf shrinkage (`SMA-EA`) improve subject-level LODO accuracy by **+3.07 pp** (**+3.84 pp** after correcting `Cho2017` MOABB loader block-concatenation) over unaligned decoding and outperform fixed-prior identity shrinkage ($\kappa = 10$) by **+0.93 pp** (Holm $p = 0.0178$).
4. **Statistical Equivalence of Feature-Space Adaptation (TOST):** Across 17 benchmarked methods, `ShallowConvNet` (**70.63% ± 0.79%** subject mean, **73.56%** macro), `EEG-Conformer` (**70.63% ± 0.80%**, **73.75%** macro), and `Deep4Net` (**70.53% ± 0.79%**, **73.72%** macro) lead. Once spatial covariances are aligned in signal space, multi-source `DANN` ($lpha_{\max} \in \{0.05, 0.15, 0.30\}$), `Deep CORAL`, `AdaBN`, and `T-TIME` all exhibit 90% CIs strictly within `[-0.16, +0.35]` pp of standard `EEGNet-8,2` (**70.00% ± 0.82%**; TOST $p < 0.001$).

---

## 2. Repository Structure

```text
Cross-Device-EEG-Benchmark/
├── src/
│   ├── alignment.py                    # Global-RMS scaling, Batch EA, Recursive Online EA (k=0) & SMA-EA
│   ├── harmonization.py                # Dense-First vs. Select-First 10-05 spherical spline harmonization
│   └── models.py                       # SimpleCNN, ShallowConvNet, Deep4Net, EEG-Conformer, EEGNet-8,2, DANN
├── tests/
│   └── test_alignment.py               # Unit tests: strict temporal causality, batch whitening & 19D CAR basis
├── scripts/
│   └── reproduce_stats_and_tables.py   # Reproduces all manuscript tables, bootstrap CIs, LME & TOST tests
├── results/                            # Precomputed subject-level & multi-seed CSV result tables
└── figures/                            # Publication figures (PDF & 300-DPI PNG)
