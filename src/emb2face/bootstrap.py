from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


DEFAULT_REPO_URL = "https://github.com/charan-v2/emb2face.git"
DEFAULT_COLAB_DIR = Path("/content/emb2face")


def is_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def in_repo_checkout(path: Optional[Path] = None) -> bool:
    base = Path(path) if path is not None else Path.cwd()
    return (base / "pyproject.toml").exists() and (base / "src" / "emb2face").exists()


def run(cmd):
    subprocess.check_call(cmd)


def install_requirements(repo_dir: Path) -> None:
    req = repo_dir / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    run([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)])


def clone_repo(repo_url: str = DEFAULT_REPO_URL, target_dir: Optional[Path] = None) -> Path:
    dest = Path(target_dir) if target_dir is not None else DEFAULT_COLAB_DIR
    if (dest / "pyproject.toml").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, str(dest)])
    return dest


def prepare_environment(
    repo_url: str = DEFAULT_REPO_URL,
    target_dir: Optional[Path] = None,
    install: bool = True,
) -> Path:
    if in_repo_checkout():
        repo_dir = Path.cwd()
    elif is_colab():
        repo_dir = clone_repo(repo_url=repo_url, target_dir=target_dir)
    else:
        repo_dir = repo_root()

    if install:
        install_requirements(repo_dir)

    src_dir = repo_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)
    return repo_dir


def default_config_path() -> Path:
    repo_dir = repo_root()
    candidate = repo_dir / "config" / "default.yaml"
    if candidate.exists():
        return candidate
    return Path("config/default.yaml")
