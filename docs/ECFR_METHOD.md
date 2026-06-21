# ECFR Method Notes

This note maps the MICCAI paper method to the released implementation.

## Core Idea

ECFR treats online test-time adaptation as routing samples in an entropy-consistency space. Each incoming test sample is diagnosed by:

- Entropy `H`: uncertainty of the teacher prediction.
- Consistency `D`: Jensen-Shannon divergence between teacher and student predictions.

The sample location determines whether ECFR minimizes or maximizes entropy and consistency. Reliable samples are consolidated, while high-risk samples are pushed away from overconfident confirmation bias.

## Paper-to-Code Map

| Paper component | Implementation |
| --- | --- |
| Strict online stream, batch size 1, no replay | `main_adaptation.py` dataloader |
| Weak and strong augmentations | `datasets/augmentations.py` |
| Student model `f_theta` | `methods/ecfr.py::ECFRAdaptation.student` |
| EMA teacher `f_theta'` | `methods/ecfr.py::ECFRAdaptation.teacher` |
| Teacher update with EMA alpha = 0.999 | `--ema_alpha`, `methods/ecfr.py` |
| Entropy `H` | `utils/metrics.py::compute_entropy` |
| JS divergence `D` | `utils/metrics.py::compute_js_divergence` |
| FIFO memory bank with capacity `N=128` | `models/memory_bank.py::DynamicMemoryBank` |
| Memory-bank threshold activation | `--warmup_size 5`, i.e., signed max updates activate after at least 5 samples are available |
| Quantile thresholds `tau_H`, `tau_D` | `DynamicMemoryBank.get_thresholds` |
| ECFR-q / ECFR-0.5 / ECFR-Acc | `--quantile` |
| Signed switches `lambda_e`, `lambda_c` | `DynamicMemoryBank.get_coefficients` |
| Quadrant-wise flow rectification loss | `methods/ecfr.py::forward_and_adapt` |
| Stochastic restoration with `p=0.01` | `--restore_prob`, `ECFRAdaptation._stochastic_restore` |
| LayerNorm affine adaptation | `--adapt_params layernorm` |
| Seven-point learning-rate sweep | `scripts/run_lr_sweep.sh` |
| Lowest-Brier learning-rate selection | `scripts/run_lr_sweep.sh` and `--output_csv` |
| Entropy-consistency scatter plot | `utils/visualization.py` |

## Optimization Rule

ECFR uses signed coefficients rather than ordinary positive loss weights:

```text
lambda_e = +1 if D < tau_D, otherwise -1
lambda_c = +1 if H < tau_H, otherwise -1
```

The released loss follows the implementation used in the experiments:

```text
L_total = lambda_e * H(p_student) + lambda_c * JS(p_teacher, p_student)
```

Teacher entropy is used for sample-state diagnosis. Student entropy is used for the optimization term, while consistency is measured between teacher and student predictions.

## ECFR Variants

`ECFR-0.5` uses:

```bash
QUANTILE=0.5
```

`ECFR-Acc` uses a scalar prior reliability estimate:

```bash
QUANTILE=<prior_target_accuracy>
```

For public benchmarks, the paper uses zero-shot accuracy only as this scalar proxy. During online adaptation, labels are not used sample by sample.

## Included Baseline

The repository also includes `methods/grata.py`, a classification-adapted GraTa baseline used for comparison in the paper. Other baselines are intentionally not included in this compact official release.
