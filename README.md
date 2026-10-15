# Camouflaged Object Detection with SAM ViT-B on COD10K

This project explores **camouflaged object detection (COD)** using the Segment Anything Model (SAM), ViT-B, and the COD10K dataset. The work is organized as a sequence of experiments, from an initial baseline through edge-aware ideas and post-processing to full-model fine-tuning.

## Experimental workflow

The notebooks are intended to be followed in this order:

1. **Baseline** — establish an initial SAM-based COD pipeline.
2. **Edge-aware experiments** — investigate edge-guided additions to SAM.
3. **Edge Pyramid Adapter** — explore multi-scale edge features.
4. **Post-processing ablations** — evaluate thresholding, test-time augmentation, self-prompting, and other inference-time changes.
5. **Final training** — fine-tune SAM ViT-B for COD10K and evaluate the resulting model.

The edge-aware and post-processing notebooks document intermediate experiments, including approaches that did not consistently improve the evaluation results. They are retained to show the experimental progression rather than presented as the final best method.

## Repository structure

```text
camouflaged-object-detection/
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01_sam_cod_baseline.ipynb
│   ├── 02_edge_aware_sam_experiments.ipynb
│   ├── 03_edge_pyramid_adapter.ipynb
│   ├── 04_sam_postprocessing_ablations.ipynb
│   └── 05_sam_vit_b_cod10k_final_training.ipynb
├── figures/                 # Selected project figures (optional)
├── results/                  # Small result tables/plots (optional)
└── requirements.txt          # Add after confirming notebook dependencies
```

## Dataset

The experiments use **COD10K**, a dataset for camouflaged object detection. Download the dataset from its official distribution/source and configure the dataset paths in the notebooks. The dataset itself is not included in this repository.

## Setup

Python dependencies and versions can vary between notebooks. After checking the imports used by the finalized notebooks, install the pinned dependencies with:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Open the notebooks in Jupyter or Google Colab. Update dataset and checkpoint paths for your environment before running cells. Training SAM ViT-B can require substantial GPU memory and storage.

## Evaluation

The experiments use camouflaged-object segmentation metrics such as Structure Measure (S\u03b1), Enhanced-alignment Measure (E\u03c6), weighted F-measure (F\u03b2^w), and mean absolute error (MAE), where implemented. **Compare metrics only when the evaluation split, image set, thresholding, and evaluation protocol match.** Results from different subsets or notebook protocols should not be treated as directly comparable.

## Results

A consolidated results table and qualitative examples will be added after the final evaluation protocol and figures are confirmed. No single model is labelled as best here until the results can be compared under the same protocol.

## Reproducibility and notes

- Set random seeds where supported and record the dataset split and checkpoint used.
- Keep large checkpoints, cached embeddings, dataset files, and generated outputs out of Git.
- Do not commit API keys, credentials, or local environment files.
- Follow the dataset and pretrained-model licenses/terms when using or redistributing them.

## Acknowledgements

This project builds on the Segment Anything Model (SAM) and the COD10K dataset. Please cite the original papers and dataset documentation in any report or presentation based on this work.
