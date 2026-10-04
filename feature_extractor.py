"""
Extracts DINOv3 features for a folder of images and saves one compressed
.safetensors file per image: [H, W, n_components], float16.

Works on any image folder. Optionally restrict to a subset with --image-list, or to
the images that see a splat chunk with --chunk-file + --colmap-dir.
"""

import argparse
import gc
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from sklearn.decomposition import PCA
from tqdm import tqdm

from dinov3_utils import get_device

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

MODELS = {
    "vit": {"default_weights": "dinov3_vit7b16.pth"},
    "convnext": {"default_weights": "dinov3_convnext_large.pth"},
}


# --- Image discovery ---


def find_images(
    images_dir: Path,
    image_list: Path | None,
    chunk_file: Path | None,
    colmap_dir: Path | None,
):
    """Returns image paths relative to images_dir."""
    if chunk_file is not None:
        if colmap_dir is None:
            raise SystemExit("--chunk-file requires --colmap-dir")
        from chunk_filter import (
            get_images_for_chunk,
        )  # your own module, only needed here

        names = list(get_images_for_chunk(str(chunk_file), str(colmap_dir)))
    elif image_list is not None:
        names = [l.strip() for l in image_list.read_text().splitlines() if l.strip()]
    else:
        names = sorted(
            str(p.relative_to(images_dir))
            for p in images_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        )

    stems = {}
    for n in names:
        stem = Path(n).stem
        if stem in stems and stems[stem] != n:
            raise SystemExit(
                f"Duplicate image name '{stem}' ({stems[stem]} and {n}). "
                "Output files are named by image stem, so these would overwrite each other."
            )
        stems[stem] = n
    return names


# --- Model setup ---


def build_extractor(args, device):
    """Returns a function Image -> [H, W, C] float16 CPU feature map."""
    weights = args.model_path or Path(MODELS[args.model]["default_weights"])

    if args.model == "vit":
        from vit_processor import load_vit_model, process_image_vit_blended

        model = load_vit_model(str(args.repo_dir), str(weights), device)
        return lambda img: process_image_vit_blended(
            model,
            img,
            device,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
        )

    from convnext_processor import (
        load_convnext_model,
        process_image_convnext_singlepass,
    )

    model = load_convnext_model(str(args.repo_dir), str(weights), device)
    return lambda img: process_image_convnext_singlepass(model, img, device)


def train_pca(
    extract, images_dir, names, n_components, n_samples, pixels_per_image, seed
):
    rng = random.Random(seed)
    subset = rng.sample(names, min(n_samples, len(names)))
    print(f"--- Training PCA on {len(subset)} images ---")

    buffer = []
    for i, name in enumerate(subset):
        print(f"PCA sample {i + 1}/{len(subset)}: {name}")
        try:
            image = Image.open(images_dir / name).convert("RGB")
            feat = extract(image)
            flat = feat.contiguous().view(-1, feat.shape[-1])
            idx = torch.randperm(flat.shape[0])[:pixels_per_image]
            buffer.append(flat[idx].float().numpy())
            del feat, flat
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {name}: {e}")

    if not buffer:
        return None
    stack = np.concatenate(buffer, axis=0)
    if stack.shape[0] < n_components:
        raise SystemExit(
            f"Only {stack.shape[0]} PCA samples for {n_components} components; "
            "raise --pca-samples or --pca-pixels, or lower --n-components."
        )
    print("Fitting PCA...")
    return PCA(n_components=n_components).fit(stack)


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features for a folder of images and compress them with PCA"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Folder of input images (searched recursively)",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="vit",
        help="vit = ViT-7B tiled/blended, convnext = ConvNeXt-L single pass",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Default: features_<model>"
    )
    parser.add_argument(
        "--n-components", type=int, default=64, help="Compressed channels per pixel"
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Save raw features without compression (large!)",
    )
    parser.add_argument(
        "--pca-path",
        type=Path,
        default=None,
        help="Load PCA from here if it exists, else train and save here. Default: pca_<model>_<n>.joblib",
    )
    parser.add_argument(
        "--pca-samples", type=int, default=5, help="Random images used to train the PCA"
    )
    parser.add_argument(
        "--pca-pixels",
        type=int,
        default=10000,
        help="Random pixels taken per PCA image",
    )
    parser.add_argument("--seed", type=int, default=0)

    sel = parser.add_argument_group("optional image selection")
    sel.add_argument(
        "--image-list",
        type=Path,
        help="Text file of image paths (relative to --images-dir), one per line",
    )
    sel.add_argument(
        "--chunk-file",
        type=Path,
        help="Only images that see this splat chunk (needs --colmap-dir and chunk_filter.py)",
    )
    sel.add_argument(
        "--colmap-dir", type=Path, help="COLMAP sparse folder, used with --chunk-file"
    )

    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("dinov3"),
        help="Local DINOv3 clone (auto-cloned if missing)",
    )
    mdl.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Weights (.pth). Default depends on --model",
    )
    mdl.add_argument("--patch-size", type=int, default=512, help="(vit) tile size")
    mdl.add_argument("--stride", type=int, default=256, help="(vit) tile stride")
    mdl.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="(vit) tiles per batch; lower if you run out of VRAM",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path(f"features_{args.model}")
    pca_path = args.pca_path or Path(f"pca_{args.model}_{args.n_components}.joblib")

    names = find_images(
        args.images_dir, args.image_list, args.chunk_file, args.colmap_dir
    )
    if not names:
        raise SystemExit(f"No images found in {args.images_dir}")
    print(f"Queue: {len(names)} images.")

    device = get_device()
    extract = build_extractor(args, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_projection = pca_mean = None
    if not args.no_pca:
        if pca_path.exists():
            print(f"Loading existing PCA from {pca_path}...")
            pca = joblib.load(pca_path)
        else:
            pca = train_pca(
                extract,
                args.images_dir,
                names,
                args.n_components,
                args.pca_samples,
                args.pca_pixels,
                args.seed,
            )
            if pca is None:
                raise SystemExit("PCA training failed on every sample image.")
            joblib.dump(pca, pca_path)
        pca_projection = torch.tensor(
            pca.components_, dtype=torch.float32
        )  # stays on CPU
        pca_mean = torch.tensor(pca.mean_, dtype=torch.float32)

    print("Extracting features...")
    for name in tqdm(names):
        save_path = output_dir / f"{Path(name).stem}.safetensors"
        if save_path.exists():
            continue
        try:
            image = Image.open(args.images_dir / name).convert("RGB")
            feats = extract(image)  # [H, W, C] float16, CPU
            H, W, C = feats.shape
            if pca_projection is not None:
                flat = feats.reshape(-1, C).float()
                feats = torch.matmul(flat - pca_mean, pca_projection.T).view(H, W, -1)
            save_file(
                {"features": feats.contiguous().to(torch.float16)}, str(save_path)
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n[ERROR] Failed on {name}: {e}")

    print(f"\nDone! Features saved to {output_dir}")


if __name__ == "__main__":
    main()
