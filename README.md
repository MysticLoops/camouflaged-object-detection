# Camouflaged Object Detection with SAM ViT-B on COD10K

A student research project exploring **camouflaged object detection (COD)** using the Segment Anything Model (SAM), ViT-B, and the COD10K dataset. The project follows an experimental progression from a baseline to edge-aware approaches, inference-time ablations, and full SAM ViT-B fine-tuning.

## Project workflow

The five notebooks live in the repository root and are intended to be read in this order:

1. **[`01_sam_cod_baseline.ipynb`](01_sam_cod_baseline.ipynb)** — initial SAM-based COD pipeline.
2. **[`02_edge_aware_sam_training.ipynb`](02_edge_aware_sam_training.ipynb)** — edge-aware SAM experiments.
3. **[`03_self_prompting_and_postprocessing.ipynb`](03_self_prompting_and_postprocessing.ipynb)** — self-prompting and inference-time post-processing experiments.
4. **[`04_edge_pyramid_and_threshold_ablation.ipynb`](04_edge_pyramid_and_threshold_ablation.ipynb)** — Edge Pyramid and threshold/ablation experiments.
5. **[`05_sam_vit_b_cod10k_final_training.ipynb`](05_sam_vit_b_cod10k_final_training.ipynb)** — full SAM ViT-B fine-tuning and final evaluation.

The edge-aware and post-processing approaches are kept as intermediate experiments, including methods that did not consistently improve all metrics. They are part of the project history, not claims that each method is an improvement.

## Qualitative predictions

The following figure is extracted from the saved final-training notebook output.

![COD10K test prediction examples](cod10k_figures_and_results/figures/05_test_prediction_examples.png)

More examples:
- [Dataset sample](cod10k_figures_and_results/figures/01_cod10k_dataset_sample.png)
- [Baseline prediction](cod10k_figures_and_results/figures/02_baseline_prediction_example.png)
- [Epoch prediction comparison](cod10k_figures_and_results/figures/03_epoch_prediction_comparison.png)
- [Epoch 4 vs. epoch 5 comparison](cod10k_figures_and_results/figures/04_qualitative_epoch4_vs_epoch5.png)
- [Original-resolution prediction examples](cod10k_figures_and_results/figures/06_original_resolution_prediction_examples.png)

## Reported results

### Full SAM ViT-B fine-tuning

The final-training notebook records the following test evaluation for the **epoch-4 checkpoint**:

| Metric | Reported value |
|---|---:|
| MAE ↓ | 0.0347 |
| S-measure (Sα) ↑ | 0.8436 |
| Adaptive E-measure ↑ | 0.8907 |
| Maximum E-measure ↑ | 0.9016 |
| Weighted F-measure (Fβᵂ) ↑ | 0.7408 |
| IoU ↑ | 0.6877 |
| Dice ↑ | 0.7729 |

**Reported setup:** train/validation/test split sizes of 2,736 / 304 / 2,026 images, with input size 1024. These are notebook-reported values; rerun the evaluation and verify the split and metric implementation before treating them as a reproducible benchmark.

### Post-processing ablation

The post-processing notebook also reports a separate evaluation on a subset of 2,026 positive images:

| Method | Sα ↑ | Eφ ↑ | Fβᵂ ↑ | MAE ↓ |
|---|---:|---:|---:|---:|
| Baseline reproduction | 0.7230 | 0.7490 | 0.5285 | 0.0621 |
| Tuned post-processing | 0.7283 | 0.7994 | 0.5830 | 0.0547 |
| Flip test-time augmentation | 0.7253 | 0.8003 | 0.5778 | 0.0550 |
| Self-prompting | 0.7323 | 0.7742 | 0.5188 | 0.0694 |
| Edge Pyramid | 0.7213 | 0.7439 | 0.4931 | 0.0684 |

These are **separate notebook-reported results**. Do not compare this ablation table directly against the full fine-tuning table unless the image set, prediction protocol, thresholding, and metric implementations are confirmed to match.

The metrics are also available as [CSV](cod10k_figures_and_results/results/sam_vit_b_epoch4_metrics.csv) and [JSON](cod10k_figures_and_results/results/sam_cod10k_experiment_summary.json). See [figure and results notes](cod10k_figures_and_results/README_FIGURES.md).

## Dataset

The experiments use **COD10K**, a dataset for camouflaged object detection. Download it from its official distribution/source and configure the dataset paths in the notebooks. The dataset is not included in this repository.

## Running the notebooks

The notebooks were developed for a Python/Jupyter or Google Colab workflow. Before running them:

1. Install the dependencies required by the specific notebook.
2. Configure COD10K image and mask paths for your environment.
3. Obtain the appropriate pretrained SAM ViT-B weights and configure the checkpoint path.
4. Run the notebook cells in order.

Training and inference at 1024-pixel input size can require substantial GPU memory and storage. A `requirements.txt` will be added after the dependencies and versions used across the notebooks have been checked; until then, do not assume a single pinned environment is available.

## Project artifacts and reproducibility

The repository includes notebook-derived qualitative figures and compact metric files. Large checkpoints, COD10K data, and cached embeddings are intentionally not included.

- Record the checkpoint, split, input size, thresholding method, and metric implementation for each evaluation.
- Use the same evaluation protocol when comparing models.
- Do not commit API keys, credentials, or local environment files.
- Follow the dataset and pretrained-model licenses and terms.

## Acknowledgements

This project builds on the Segment Anything Model (SAM) and the COD10K dataset. Cite the original SAM paper and COD10K dataset/paper in reports and presentations based on this work.
