# MEMO Reproduction and JS-Consistency

Test-time adaptation project for the Deep Learning 2024 course at the
University of Trento. The project reproduces MEMO (Zhang et al.,
NeurIPS 2022) on ResNet-50 and DenseNet-121, evaluated on ImageNet-A
and ImageNet-V2, and proposes JS-Consistency, a modification that adds a
Jensen-Shannon consistency term to the MEMO entropy loss.

## Method

MEMO adapts a pretrained model to each test image by minimizing the
entropy of the average prediction over B = 32 augmented views.
JS-Consistency adds a symmetric, bounded consistency penalty between each
view and the consensus prediction:

L = H(p_bar) + lambda * (1/B) * sum_i JSD(p_i || p_bar)

Setting lambda = 0 recovers MEMO exactly. The Jensen-Shannon Divergence is
used instead of KL divergence because it is symmetric and bounded in
[0, log 2], which avoids gradient spikes when the consensus assigns
near-zero probability to a class.

## Results

Top-1 accuracy (%). Best MEMO variant against best JS-Consistency variant.

| Architecture | Dataset | Baseline | MEMO | JS-Consistency |
| :--- | :--- | :---: | :---: | :---: |
| ResNet-50 | ImageNet-A | 14.27 | 23.79 | 24.23 |
| ResNet-50 | ImageNet-V2 | 69.92 | 78.81 | 78.91 |
| DenseNet-121 | ImageNet-A | 2.13 | 11.52 | 11.15 |
| DenseNet-121 | ImageNet-V2 | 61.98 | 73.48 | 73.63 |

JS-Consistency improves on MEMO in three of four combinations. On
DenseNet-121 / ImageNet-A, MEMO with flip augmentation remains best.
Adaptive BatchNorm is the largest single contributor in every setting.

## Running the notebook

The notebook was developed on Google Colab with a T4 GPU. It reads the
datasets from Google Drive at `MyDrive/DL/Datasets/`:

- `imagenet-a/`
- `imagenetv2-matched-frequency-format-val/`

To reuse the precomputed results, copy the two files from `results/` into
the same Drive folder before running. Without them, all experiments rerun
from scratch, which takes several hours. Section 3.3 of the notebook
describes both options.

## References

Zhang, M., Levine, S., & Finn, C. (2022). MEMO: Test Time Robustness via
Adaptation and Augmentation. NeurIPS 2022.
