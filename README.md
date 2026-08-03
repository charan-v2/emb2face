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
- `scripts/run_residual_mlp_sweep.py`: convenience wrapper for residual MLP hidden-dim sweeps
- `config/default.yaml`: default config you can copy and edit
- `notebooks/01_adapter_training.ipynb`: thin notebook wrapper for training
- `notebooks/02_inversion_attack_eval.ipynb`: thin notebook wrapper for attack evaluation
- `notebooks/03_colab_inference.ipynb`: Colab-ready Arc2Face inference notebook

## Requirements

- Python 3.10 or newer
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

To train the new residual MLP adapter, set `adapter_type: residual_mlp` in a config file and choose `hidden_dim` as `1024` or `2048`. For the two-run sweep, use:

```bash
python scripts/run_residual_mlp_sweep.py --config config/default.yaml
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

Inference sampling is now plain random selection. The run will still extract faces for embedding and reconstruction, but it no longer does a separate yaw-based prefilter step.

If you fill the input/output/checkpoint values in `config/default.yaml`, you can run the whole inference stage with just:

```bash
python -m emb2face infer --config config/default.yaml
```

And then score the latest inference run with:

```bash
python -m emb2face score --config config/default.yaml
```

For the GPU workflow you described, a good starting command is:

```bash
python -m emb2face infer \
  --config config/default.yaml \
  --input-dir /path/to/dataset_root \
  --output-dir /path/to/inference_runs \
  --num-identities 2000 \
  --images-per-identity 5 \
  --inference-adapter-checkpoints /path/to/best_linear_adapter.pt,/path/to/best_residual_mlp_adapter.pt \
  --device cuda
```

This step only generates reconstructions and run metadata. Similarity and biometric metrics are computed later from the saved run folder.

6. Run the scoring pipeline on a previous inference run folder.

```bash
python -m emb2face score --config config/default.yaml --input-run-dir /path/to/inference_run
```

By default the scoring pipeline uses `retinaface` for detection/alignment and `uniface` for embeddings. You can override both:

```bash
python -m emb2face score --config config/default.yaml --input-run-dir /path/to/inference_run --score-detector-backend insightface --score-embedder-backend insightface
```

If you use the default scoring backend, install UniFace first with `pip install uniface[cpu]` or `pip install uniface[gpu]`.

The scoring step writes:

- `verification_eval.csv`: FAR, FRR, FMR, FNMR, EER, and threshold summary per reconstruction method
- `verification_scores.csv`: flattened pairwise verification scores
- `det_curve.csv`: DET curve points
- `det_curve.png`: quick visual check of the DET curve

7. Run the Arc2Face inference pipeline on a single image and save the comparison panel.

```bash
python -m emb2face infer --config config/default.yaml --input-image /path/to/image.jpg
```

If you want inference to use a specific adapter checkpoint, set `inference_adapter_checkpoint` in the config to the exact `.pt` file path. To run multiple adapters in one pass, use `inference_adapter_checkpoints` with a comma-separated list on the CLI or a list in the config. For example:

```yaml
inference_adapter_checkpoints:
  - outputs/webface_arcada_adapter/models_full/best_linear_adapter.pt
  - outputs/webface_arcada_adapter/models_full/best_residual_mlp_adapter.pt
```

8. Or do both in one go.

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

## Docker

This repo includes a GPU-ready [`Dockerfile`](./Dockerfile).

For a full end-to-end GPU checklist, see [GPU_RUNBOOK.md](/Users/charan/Projects/OVGU/emb2face/GPU_RUNBOOK.md).

Build it from the repository root:

```bash
docker build -t emb2face:gpu .
```

Run inference on a machine with NVIDIA Container Toolkit enabled:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/emb2face" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$HOME/.cache/torch:/cache/torch" \
  -v "$HOME/.insightface:/root/.insightface" \
  -w /workspace/emb2face \
  emb2face:gpu \
  python -m emb2face infer --config config/default.yaml --input-dir /workspace/emb2face/data/webface_112x112 --device cuda
```

For a single image, swap `--input-dir` for `--input-image`.

If you want to use a saved adapter checkpoint, add:

```bash
--inference-adapter-checkpoint /workspace/emb2face/outputs/.../best_*.pt
```

### GPU PC Notes

The GPU-PC handout says the machine is reachable only from the university network, usually through VPN, and that work is run inside Docker containers.

For the OVGU GPU PC workflow:

1. Connect to the university VPN if needed.
2. SSH into `gensynth.cs.uni-magdeburg.de`.
3. Launch the assigned container, for example with `sudo Username.docker` or `sudo Username.docker help`.
4. Keep your repo and writable files under your mounted home directory. The handout says the host path is `/vol2/Username/` and the same directory appears inside the container as `Username/`.
5. Use shared datasets from `/vol2/share` on the host or `/share` inside the container.

If you place this repo inside your mounted home directory, you can run the same inference command with paths adjusted to the container mounts. A typical GPU-PC command looks like:

```bash
python -m emb2face infer \
  --config config/default.yaml \
  --input-dir /share/webface_112x112 \
  --output-dir /vol1/Username/emb2face_outputs/inference \
  --device cuda
```

The important path choices are:

- `dataset_root` or `--input-dir` should point at the mounted dataset path.
- `output_root` or `--output-dir` should point at a writable location under `/vol1/Username/`.
- `insight_root` should be a writable cache directory if you want the InsightFace model files to persist between runs.

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
  - `inference_report.csv`: per-image reconstruction paths and metadata
  - `summary.csv`: aggregate sampling / generation stats for the sampled set
- `inference_<runmode>/<run_id>/biometric_eval/`: biometric scores for a previous inference run
  - `verification_eval.csv`: FAR, FRR, FMR, FNMR, EER, and threshold summary
  - `verification_scores.csv`: pairwise score table
  - `det_curve.csv`: DET curve data
  - `det_curve.png`: DET curve plot

## Notes

- The attack pipeline needs the adapter checkpoint produced by the training pipeline.
- The evaluation workflow expects the same identity split logic used during training.
- The face models used by the pipeline are downloaded on demand and cached locally.
