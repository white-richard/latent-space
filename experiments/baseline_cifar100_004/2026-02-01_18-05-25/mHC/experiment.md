# Experiment: baseline_cifar100_mHC
**Project:** [[CIFAR Lightning]]
**Date:** 2026-02-01
**Status:** #training-complete

## 🧪 Hypothesis
mHC improves representation quality without slowing convergence.

## 🛠 Parameters & Setup
- **Hardware:** RTX 3090
- **Model:** vit_small
- **Learning Rate:** 0.001
- **Batch Size:** 256
- **Full Config:** ![[config.md]]

## 📊 Results & Plots
- **Final Metrics:**
| test/loss | test/acc | test/knn_acc | test/silhouette |
| --- | --- | --- | --- |
| 4.69116 | 0.0078125 | 0.0622 | -0.231851 |

## 📝 Observations & Conclusions
- [Note on convergence speed]
- [Unforeseen issues]
- **Conclusion:** [Was the hypothesis proven?]
