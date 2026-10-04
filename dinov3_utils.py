"""Shared helpers for the DINOv3 feature-extraction scripts."""

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.transforms import v2

DINOV3_GIT_URL = "https://github.com/facebookresearch/dinov3.git"


# --- Repo / model loading ---


def ensure_dinov3_repo(repo_dir: str) -> None:
    """Clones the DINOv3 repo into repo_dir if it isn't already there."""
    repo = Path(repo_dir)
    if (repo / "hubconf.py").exists():
        return
    print(f"DINOv3 repo not found at {repo}, cloning from {DINOV3_GIT_URL} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", DINOV3_GIT_URL, str(repo)], check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"Could not clone DINOv3 automatically ({e}). Install git, or run:\n"
            f"  git clone {DINOV3_GIT_URL} {repo}"
        ) from e


def load_hub_model(
    hub_name: str,
    repo_dir: str,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """
    Loads a DINOv3 model (e.g. "dinov3_vit7b16", "dinov3_convnext_large") from a
    local repo checkout + local weights, optionally casting to `dtype` on the CPU
    before moving to `device`.
    """
    print(f"Loading {hub_name} from: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Weights not found at {model_path}. The DINOv3 weights are gated: request "
            "access on Meta's DINOv3 download page, then download the .pth file manually."
        )
    ensure_dinov3_repo(repo_dir)

    # The DINOv3 hub script loads to CPU first
    model = torch.hub.load(repo_dir, hub_name, source="local", weights=model_path)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.to(device)
    model.eval()
    print("Model loaded successfully to device.")
    return model


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# --- Feature files ---


def infer_feature_dim(features_dir) -> int:
    """Reads the channel count from the first .safetensors file in features_dir."""
    from safetensors import safe_open

    first = next(Path(features_dir).glob("*.safetensors"), None)
    if first is None:
        raise FileNotFoundError(
            f"No .safetensors feature files found in {features_dir}"
        )
    with safe_open(str(first), framework="pt") as f:
        key = next(iter(f.keys()))
        dim = f.get_slice(key).get_shape()[-1]
    print(f"Detected feature dim {dim} from {first.name}")
    return int(dim)


# --- Preprocessing ---


def build_transform() -> v2.Compose:
    """ImageNet-normalised tensor transform (no resize/crop)."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


# --- CLI ---


def add_common_args(
    parser: argparse.ArgumentParser, default_model_path: str, default_output: str
) -> None:
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("dinov3"),
        help="Local clone of facebookresearch/dinov3 (auto-cloned if missing)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(default_model_path),
        help="Path to weights (.pth)",
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.add_argument(
        "--no-show", action="store_true", help="Save the plot without opening a window"
    )


# --- Visualisation ---


def pca_to_rgb(feature_map: torch.Tensor) -> np.ndarray:
    """[H, W, C] feature map -> [H, W, 3] uint8 image via 3-component PCA."""
    h, w, c = feature_map.shape
    feats = feature_map.to(torch.float32).reshape(-1, c).numpy()
    projected = PCA(n_components=3).fit_transform(feats)
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    normalised = (projected - lo) / (hi - lo)
    return (normalised.reshape(h, w, 3) * 255).astype(np.uint8)


def plot_pca_comparison(
    image: Image.Image,
    feature_map: torch.Tensor,
    model_stride: int,
    title: str,
    output: Path,
    show: bool = True,
) -> None:
    """Saves (and optionally shows) original vs. upsampled PCA of the features."""
    print("Applying PCA to full feature map for visualization...")
    final_h, final_w, _ = feature_map.shape
    pca_image = Image.fromarray(pca_to_rgb(feature_map))

    target_size = (final_w * model_stride, final_h * model_stride)
    print(
        f"Upsampling PCA image from {final_w}x{final_h} to {target_size[0]}x{target_size[1]}..."
    )
    upsampled = pca_image.resize(target_size, Image.Resampling.BILINEAR)

    # Crop the original to the feature map's extents so both panels line up
    cropped = image.crop((0, 0, target_size[0], target_size[1]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_title(f"Original Image (cropped to {target_size[0]}x{target_size[1]})")
    axes[0].imshow(cropped)
    axes[0].axis("off")

    axes[1].set_title(title)
    axes[1].imshow(upsampled)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output)
    if show:
        print("Displaying final results... Close the plot window to exit.")
        plt.show()
    print(f"Saved '{output}'.")
