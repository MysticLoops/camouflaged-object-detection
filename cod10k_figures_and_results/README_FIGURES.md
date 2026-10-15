# COD10K Figures and Results

These figures were extracted from the saved outputs embedded in the uploaded `05_sam_vit_b_cod10k_final_training.ipynb`. They are the notebook's own outputs, not newly fabricated illustrations.

## Figures
- `01_cod10k_dataset_sample.png`: dataset sample visualization (notebook cell 3).
- `02_baseline_prediction_example.png`: early prediction example (cell 26).
- `03_epoch_prediction_comparison.png`: epoch prediction visualization (cell 71).
- `04_qualitative_epoch4_vs_epoch5.png`: epoch 4 and epoch 5 qualitative comparison (cell 72).
- `05_test_prediction_examples.png`: test prediction examples (cell 98).
- `06_original_resolution_prediction_examples.png`: original-resolution prediction visualization (cell 100).

## Results
`results/sam_vit_b_epoch4_metrics.csv` and `results/sam_cod10k_experiment_summary.json` contain metrics transcribed from the notebook's saved summary output (cell 102 / cell 97 output):
- Model: SAM ViT-B
- Dataset: COD10K
- Selected checkpoint: epoch 4
- Input size: 1024
- Split sizes: 2,736 train / 304 validation / 2,026 test

These are reported notebook results. Re-run the evaluation before making benchmark claims, and ensure the metric implementation and split protocol are documented. Do not compare these values directly with the separate post-processing ablation notebook unless the evaluation protocols match.
