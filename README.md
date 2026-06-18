# emb2face

Pipeline for learning an adapter between AdaFace and ArcFace spaces, then running the inversion-attack evaluation.

The repository is organized so the real logic lives in Python modules under `src/emb2face/`, while the notebooks stay thin and only call the CLI.

## What you get

- `train`: extracts paired embeddings, trains the adapter, and writes the split metadata.
- `attack`: loads the trained adapter and runs the inversion attack experiments.
- `all`: runs training first, then the attack pipeline.

## Project layout

- `src/emb2face/cli.py`: command-line entrypoint
- `src/emb2face/config.py`: config loading and output-path setup
- `src/emb2face/embeddings.py`: dataset scan, face loading, and embedding extraction
- `src/emb2face/train.py`: adapter training pipeline
- `src/emb2face/attack.py`: inversion attack pipeline
- `config/default.yaml`: default config you can copy and edit
- `notebooks/01_adapter_training (2).ipynb`: thin notebook wrapper for training
- `notebooks/02_inversion_attack_eval (2).ipynb`: thin notebook wrapper for attack evaluation

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
git clone <your-repo-url>
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

5. Or do both in one go.

```bash
python -m emb2face all --config config/default.yaml
```

## Google Colab Run

There are two easy ways to use Colab:

### Option 1: Run the thin notebooks

1. Open the notebook from the repo in Colab.
2. Mount Drive if you want to keep datasets and outputs there.
3. Install the repo into the runtime before the notebook cell that calls `main(...)`.

Typical Colab bootstrap:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone <your-repo-url> /content/emb2face
%cd /content/emb2face
!pip install -r requirements.txt
!pip install -e .
```

Then run:

```python
from emb2face.cli import main
main(['train', '--config', 'config/default.yaml'])
```

or for the attack notebook:

```python
from emb2face.cli import main
main(['attack', '--config', 'config/default.yaml'])
```

### Option 2: Run the CLI directly in Colab

After the same bootstrap above, you can use:

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

## Notes

- The attack pipeline needs the adapter checkpoint produced by the training pipeline.
- The evaluation workflow expects the same identity split logic used during training.
- The face models used by the pipeline are downloaded on demand and cached locally.

