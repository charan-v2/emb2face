# GPU Runbook

This file is the end-to-end checklist for running `emb2face` on the OVGU GPU PC or any similar Linux GPU host with Docker support.

## 1. Prepare the local layout

Keep the same structure locally and on the GPU machine if possible:

```text
emb2face/
├── data/
├── outputs/
├── config/
├── scripts/
└── src/
```

The repo already ignores `data/` and `outputs/`, so you can keep datasets, checkpoints, and generated results inside the repo without committing them.

## 2. Copy the project to the GPU machine

If you want the full local state, copy the repo from your laptop to the shared directory on the GPU machine:

```bash
scp -r /Users/charan/Projects/OVGU/emb2face \
  pitsec_sose26_topic9@gensynth.cs.uni-magdeburg.de:/vol2/pitsec_sose26_topic9/sharedDockerDir/
```

After that, the repo lives at:

```bash
/vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face
```

If you prefer a fresh clone on the GPU PC instead:

```bash
ssh pitsec_sose26_topic9@gensynth.cs.uni-magdeburg.de
cd /vol2/pitsec_sose26_topic9/sharedDockerDir
git clone <your-repo-url> emb2face
```

## 3. Put data and outputs in shared storage

Recommended locations:

- Repo: `/vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face`
- Data: `/vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/data`
- Outputs: `/vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs`
- Caches: `/vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/.cache`

The GPU handout says the host home directory is mounted into the container, and shared datasets are available from `/vol1/share` on the host or `/share` inside the container. On this project, the simplest path is to keep everything inside the shared repo folder unless your course setup gives you a separate dataset mount.

## 4. SSH to the GPU machine and start Docker

From your local machine, connect to the host:

```bash
ssh pitsec_sose26_topic9@gensynth.cs.uni-magdeburg.de
```

If the GPU PC requires VPN access, connect to the university VPN first.

Then start the assigned container using the course command, for example:

```bash
sudo pitsec_sose26_topic9.docker
```

If you need help listing containers:

```bash
sudo pitsec_sose26_topic9.docker help
```

## 5. Install dependencies in the container

Once inside the container, go to the repo:

```bash
cd /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face
```

Install everything in one shot:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

That includes:

- the core Python stack
- GPU PyTorch
- `onnxruntime-gpu` on Linux
- `uniface[gpu]` for scoring on Linux

The first run may still download model weights and face-model caches automatically.

## 6. Sanity-check the GPU

Inside the container:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

You want `torch.cuda.is_available()` to print `True`.

## 7. Configure paths

You can keep using `config/default.yaml` and override paths on the command line, or edit the config file for the GPU machine.

Suggested values:

```yaml
dataset_root: /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/data/webface_112x112
output_root: /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter
device: cuda
insight_root: /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/.insightface
arc2face_local_dir: /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/.cache/arc2face_models
```

If you already have a trained adapter checkpoint, set:

```yaml
inference_adapter_checkpoint: /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/models_full/best_residual_mlp_adapter.pt
```

If you want to run both adapters in the same inference pass, use:

```yaml
inference_adapter_checkpoints:
  - /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/models_full/best_linear_adapter.pt
  - /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/models_full/best_residual_mlp_adapter.pt
inference_max_yaw_degrees: 45
inference_pose_require_single_face: true
```

## 8. Run inference

Folder inference:

```bash
python -m emb2face infer \
  --config config/default.yaml \
  --input-dir /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/data/webface_112x112 \
  --output-dir /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/inference_full \
  --device cuda
```

Single-image inference:

```bash
python -m emb2face infer \
  --config config/default.yaml \
  --input-image /path/to/image.jpg \
  --output-dir /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/inference_full \
  --device cuda
```

If you want to use a specific checkpoint:

```bash
python -m emb2face infer \
  --config config/default.yaml \
  --input-dir /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/data/webface_112x112 \
  --output-dir /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/inference_full \
  --inference-adapter-checkpoints /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/models_full/best_linear_adapter.pt,/vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/models_full/best_residual_mlp_adapter.pt \
  --num-identities 2000 \
  --images-per-identity 5 \
  --inference-max-yaw-degrees 45 \
  --inference-pose-require-single-face \
  --device cuda
```

The run writes a pose-filter manifest alongside the normal inference outputs, and the scoring command stays the same because it already picks up any `*_recon_path` / `*_recon_paths` columns in the inference report.

If you keep the GPU defaults in `config/default.yaml`, the shortest form is:

```bash
python -m emb2face infer --config config/default.yaml
python -m emb2face score --config config/default.yaml
```

## 9. Run scoring

Score a previous inference run by pointing to the `run_*` folder created by inference:

```bash
python -m emb2face score \
  --config config/default.yaml \
  --input-run-dir /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/inference_full/run_YYYYMMDD_HHMMSS_xxxxxxxx \
  --device cuda
```

If you want to be explicit about the scoring backends:

```bash
python -m emb2face score \
  --config config/default.yaml \
  --input-run-dir /vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs/webface_arcada_adapter/inference_full/run_YYYYMMDD_HHMMSS_xxxxxxxx \
  --score-detector-backend insightface \
  --score-embedder-backend insightface \
  --device cuda
```

The score run writes the biometric outputs under a `biometric_eval/` folder inside the run directory.

## 10. Copy results back if needed

If everything is already in the shared directory, you may not need to copy anything.

If you want to bring results back to your laptop:

```bash
scp -r pitsec_sose26_topic9@gensynth.cs.uni-magdeburg.de:/vol2/pitsec_sose26_topic9/sharedDockerDir/emb2face/outputs \
  /Users/charan/Projects/OVGU/emb2face/
```

## 11. Docker reference

The repo includes a `Dockerfile` that can be used to build a GPU image on a normal NVIDIA Docker host:

```bash
docker build -t emb2face:gpu .
```

Example run command:

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

On the OVGU GPU PC, the course-provided container launcher is usually the right way to start the image, and the mounted `/vol2/.../sharedDockerDir` path should be visible inside the container.
