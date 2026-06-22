# emb2face

Pipeline for learning an adapter between AdaFace and ArcFace spaces, then running the inversion-attack evaluation.

The repository is organized so the real logic lives in Python modules under `src/emb2face/`, while the notebooks stay thin and bootstrap the environment before calling the CLI.

## What you get

- `train`: extracts paired embeddings, trains the adapter, and writes the split metadata.
- `attack`: loads the trained adapter and runs the inversion attack experiments.
- `infer`: samples identities/images from a dataset-style folder and runs Arc2Face reconstruction.
- `all`: runs training first, then the attack pipeline.

## Project layout

- `src/emb2face/cli.py`: command-line entrypoint
- `src/emb2face/config.py`: config loading and output-path setup
- `src/emb2face/embeddings.py`: dataset scan, face loading, and embedding extraction
- `src/emb2face/train.py`: adapter training pipeline
- `src/emb2face/attack.py`: inversion attack pipeline
- `config/default.yaml`: default config you can copy and edit
- `notebooks/01_adapter_training.ipynb`: thin notebook wrapper for training
- `notebooks/02_inversion_attack_eval.ipynb`: thin notebook wrapper for attack evaluation

## Requirements

- Python 3.9 or newer
- A CUDA GPU is strongly recommended for the attack pipeline
- Access to the face datasets and model downloads used by the pipeline

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
pip install -e .
```

## Local Run

1. Clone the repo and enter it.

```bash
git clone https://github.com/charan-v2/emb2face
cd emb2face
```

2. Edit `config/default.yaml` for your machine.

Key paths to set:

- `dataset_root`: directory with identity subfolders for the adapter training data
- `output_root`: where embeddings, models, and reports should be written
- `eval_dataset_root`: only needed if `eval_source: external`
- `device`: use `auto`, `cuda`, `mps`, or `cpu`

3. Train the adapter.

```bash
python -m emb2face train --config config/default.yaml
```

4. Run the attack/evaluation pipeline.

```bash
python -m emb2face attack --config config/default.yaml
```

5. Run the Arc2Face inference pipeline on a folder of identities.

```bash
python -m emb2face infer --config config/default.yaml --input-dir /path/to/dataset_root
```

To sample a presentation set, pick identities first and then images per identity:

```bash
python -m emb2face infer --config config/default.yaml --input-dir /path/to/dataset_root --num-identities 10 --images-per-identity 1 --save-comparison-figures
```

6. Run the Arc2Face inference pipeline on a single image and save the comparison panel.

```bash
python -m emb2face infer --config config/default.yaml --input-image /path/to/image.jpg
```

7. Or do both in one go.

```bash
python -m emb2face all --config config/default.yaml
```

## Google Colab Run

There are two easy ways to use Colab:

### Option 1: Run the thin notebooks

1. Open the notebook from the repo in Colab or locally.
2. Run the first cell.
3. The notebook will:
   - locate the repo if it is already checked out locally
   - clone the repo into `/content/emb2face` on Colab if needed
   - install dependencies
   - mount Drive on Colab
   - launch the selected pipeline

### Option 2: Run the CLI directly in Colab

If you prefer shell commands after the notebook bootstrap has run, you can still use:

```python
!python -m emb2face train --config config/default.yaml
!python -m emb2face attack --config config/default.yaml
```

## Recommended Colab paths

If you are using Google Drive:

- `dataset_root`: `/content/drive/MyDrive/webface_112x112`
- `output_root`: `/content/drive/MyDrive/webface_arc_ada_adapter`
- `eval_dataset_root`: `/content/drive/MyDrive/celeb_eval`

You can also keep `config/default.yaml` in Drive and pass that path to the CLI.

## Outputs

The pipeline writes artifacts under `output_root`, grouped by run mode:

- `models_<runmode>/`: trained adapter checkpoints
- `reports_<runmode>/`: CSV metrics, plots, and split metadata
- `embeddings_<runmode>/`: saved source embeddings and checkpoints
- `attack_<runmode>/`: evaluation embeddings, reconstructions, and reports
- `inference_<runmode>/`: sampled inference reconstructions, comparison figures, and CSV reports
  - `selected_samples.csv`: sampled identities/images
  - `inference_report.csv`: per-image reconstruction scores and paths
  - `summary.csv`: aggregate metrics for the sampled set

## Notes

- The attack pipeline needs the adapter checkpoint produced by the training pipeline.
- The evaluation workflow expects the same identity split logic used during training.
- The face models used by the pipeline are downloaded on demand and cached locally.
