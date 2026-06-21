# ECFR

Official implementation of **ECFR: Entropy-Consistency Flow Rectification in Test-Time Adaptation for Medical Foundation Models**.

ECFR performs online test-time adaptation by routing streaming samples in an entropy-consistency space. The implementation includes the ECFR adaptation module, the classification-adapted GraTa baseline used in our comparison, metric utilities, and visualization code for entropy-consistency flow analysis.

For a paper-to-code map of the ECFR mechanism, see [docs/ECFR_METHOD.md](docs/ECFR_METHOD.md).

## Repository Structure

```text
ECFR/
|-- datasets/
|   |-- augmentations.py
|   `-- build_dataset.py
|-- docs/
|   `-- ECFR_METHOD.md
|-- methods/
|   |-- ecfr.py
|   `-- grata.py
|-- models/
|   `-- memory_bank.py
|-- scripts/
|   |-- run_adaptation.sh
|   `-- run_lr_sweep.sh
|-- utils/
|   |-- metrics.py
|   `-- visualization.py
|-- main_adaptation.py
|-- requirements.txt
`-- README.md
```

## Installation

```bash
conda create -n ecfr python=3.10
conda activate ecfr
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default `pip` command does not match your environment.

## Included Methods

This repository focuses on the official ECFR release and the classification-adapted GraTa comparison used in the paper.

| Method | File | Note |
| --- | --- | --- |
| ECFR | `methods/ecfr.py` | Main proposed method with entropy-consistency routing, dynamic quantile thresholds, EMA teacher, and stochastic restoration. |
| GraTa-classification | `methods/grata.py` | Classification adaptation of GraTa, originally proposed for medical image segmentation. |

Other baselines from the paper are not included in this compact release.

## Data Format

The adaptation entry point expects a CSV file with at least two columns:

```csv
filepath,label
/path/to/image_001.png,0
/path/to/image_002.png,1
```

The loader reads images in streaming order and returns clean, weakly augmented, and strongly augmented views for each sample.

## Model and Dataset Resources

The paper evaluates medical foundation models and public medical image datasets. For reproducibility, download model weights and datasets from their official sources and place them outside the repository.

| Resource | Link |
| --- | --- |
| Ark / Ark+ | https://github.com/JLiangLab/Ark |
| DermLIP / Derm1M | https://github.com/SiyuanYan1/Derm1M |
| Kermany OCT/CXR dataset (Mendeley v2; CXR subset used) | https://data.mendeley.com/datasets/rscbjbr9sj/2 |
| Daffodil / Skin Disease Classification Dataset | https://data.mendeley.com/datasets/3hckgznc67/1 |
| HAM10000 | https://doi.org/10.1038/sdata.2018.161 |

Large datasets, checkpoints, logs, and generated results should not be committed to Git.

## Paper Defaults

| Setting | Default |
| --- | --- |
| Streaming batch size | 1 |
| Replay buffer | None |
| Memory bank capacity | 128 |
| Memory-bank threshold activation | after at least 5 samples are available |
| EMA teacher momentum | 0.999 |
| Stochastic restoration probability | 0.01 |
| Adapted parameters | LayerNorm affine parameters |
| Optimizer | SGD with momentum 0.9 |
| Learning-rate candidates | `1e-5 3.3e-5 6.6e-5 1e-4 3.3e-4 6.6e-4 1e-3` |
| Selection criterion | Lowest Brier score |

## Learning-Rate Selection

For paper-style reporting, do not assume a fixed learning rate such as `1e-4`.
We use the predefined learning-rate set

```bash
1e-5 3.3e-5 6.6e-5 1e-4 3.3e-4 6.6e-4 1e-3
```

and select the run with the lowest Brier score.

Example for ECFR-0.5:

```bash
DATASET_CSV=/path/to/test.csv METHOD=ecfr QUANTILE=0.5 bash scripts/run_lr_sweep.sh
```

Example for ECFR-Acc:

```bash
DATASET_CSV=/path/to/test.csv METHOD=ecfr QUANTILE=0.83 bash scripts/run_lr_sweep.sh
```

Here `QUANTILE=0.83` is only an example. For ECFR-Acc, set `QUANTILE` to the prior target accuracy as a numerical value in `[0, 1]`. The command-line argument is still `--quantile`; do not pass the string `Acc`.

Run the classification-adapted GraTa baseline with the same learning-rate sweep:

```bash
DATASET_CSV=/path/to/test.csv METHOD=grata bash scripts/run_lr_sweep.sh
```

## Single Run

After selecting the learning rate from the sweep, a single run can be launched as:

```bash
DATASET_CSV=/path/to/test.csv METHOD=ecfr LR=<selected_lr> QUANTILE=0.5 bash scripts/run_adaptation.sh
```

For ECFR-Acc, replace `QUANTILE=0.5` with the prior target accuracy.

For Ark+-style high-resolution preprocessing, set `INPUT_SIZE=768`. For demonstration backbones without LayerNorm, set `ADAPT_PARAMS=norm`; paper experiments use `ADAPT_PARAMS=layernorm`.

## Notes

`main_adaptation.py` provides a compact adaptation entry point. For full reproduction of paper results, replace the demo backbone loader with the official Ark+ or DermLIP checkpoint loader, and use the same preprocessing, dataset splits, and learning-rate selection protocol described in the paper.

The visualization utility in `utils/visualization.py` can generate entropy-consistency scatter plots with threshold lines and optional error coloring.

## License

This project is released under the MIT License.
