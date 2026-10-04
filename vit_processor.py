import argparse
import math

import torch
from PIL import Image

from dinov3_utils import (
    add_common_args,
    build_transform,
    get_device,
    load_hub_model,
    plot_pca_comparison,
)

MODEL_PATCH_SIZE = 16  # Fixed for this model


def load_vit_model(repo_dir: str, model_path: str, device: torch.device):
    """Loads the DINOv3 ViT-7B model in bfloat16 to the specified device."""
    return load_hub_model(
        "dinov3_vit7b16", repo_dir, model_path, device, dtype=torch.bfloat16
    )


def process_image_vit_blended(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 256,
    batch_size: int = 8,
):
    """
    Processes a large PIL Image using a tile-and-blend strategy
    to create a seamless, high-resolution feature map.

    Returns:
        torch.Tensor: The final seamless feature map, cropped to the
                      original image size, on the CPU in float16.
    """

    # --- A. Setup Parameters ---
    c_feat = 4096  # Fixed for this model
    FEATURES_PER_PATCH = patch_size // MODEL_PATCH_SIZE

    if stride > patch_size:
        raise ValueError("Stride must be <= Patch Size for blending.")

    print(
        f"Starting blended processing with patch_size={patch_size}, stride={stride}, batch_size={batch_size}"
    )

    img_w, img_h = image.size

    patch_transform = build_transform()

    # --- B. Create "Pyramid" Weight Mask (on GPU) ---
    # Sample at cell centres so the weight never hits exactly 0 (otherwise features
    # at the outer image border would only ever receive zero weight).
    centres = (torch.arange(FEATURES_PER_PATCH) + 0.5) / FEATURES_PER_PATCH * 2 - 1
    tent_1d = (1 - torch.abs(centres)).to(device)
    weight_mask_2d = tent_1d.unsqueeze(1) * tent_1d.unsqueeze(0)
    weight_mask = weight_mask_2d.unsqueeze(0).unsqueeze(-1)  # Shape: [1, FEAT, FEAT, 1]

    # --- C. Pad Image and Create CPU Accumulator Buffers ---
    n_patches_x = math.ceil((img_w - patch_size) / stride) + 1
    n_patches_y = math.ceil((img_h - patch_size) / stride) + 1
    pad_w = (n_patches_x - 1) * stride + patch_size
    pad_h = (n_patches_y - 1) * stride + patch_size

    padded_image = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
    padded_image.paste(image, (0, 0))

    feat_h_padded = (n_patches_y - 1) * (
        stride // MODEL_PATCH_SIZE
    ) + FEATURES_PER_PATCH
    feat_w_padded = (n_patches_x - 1) * (
        stride // MODEL_PATCH_SIZE
    ) + FEATURES_PER_PATCH

    # Use float32 for precision, which is critical for good blending
    feature_accumulator = torch.zeros(
        (feat_h_padded, feat_w_padded, c_feat), dtype=torch.float32, device="cpu"
    )
    weight_accumulator = torch.zeros(
        (feat_h_padded, feat_w_padded, 1), dtype=torch.float32, device="cpu"
    )

    # --- D. Generate Patches and Coordinates ---
    patch_batches = []
    coord_batches = []
    current_patch_batch = []
    current_coord_batch = []

    print("Generating patch batches...")
    for y in range(0, pad_h - patch_size + 1, stride):
        for x in range(0, pad_w - patch_size + 1, stride):
            patch = padded_image.crop((x, y, x + patch_size, y + patch_size))
            current_patch_batch.append(patch_transform(patch))

            feat_y = y // MODEL_PATCH_SIZE
            feat_x = x // MODEL_PATCH_SIZE
            current_coord_batch.append((feat_y, feat_x))

            if len(current_patch_batch) == batch_size:
                patch_batches.append(torch.stack(current_patch_batch))
                coord_batches.append(current_coord_batch)
                current_patch_batch = []
                current_coord_batch = []

    if len(current_patch_batch) > 0:
        patch_batches.append(torch.stack(current_patch_batch))
        coord_batches.append(current_coord_batch)
    print(f"Created {len(patch_batches)} batches.")

    # --- E. Extraction, Blending, and Accumulation ---
    print("Starting batch extraction and weighted blending...")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for i, (batch, coords) in enumerate(zip(patch_batches, coord_batches)):
                # print(f"  Processing batch {i+1} / {len(patch_batches)}...")
                batch = batch.to(device)

                features_dict = model.forward_features(batch)
                feature_map_batch = features_dict["x_norm_patchtokens"]

                feature_map_4d = feature_map_batch.reshape(
                    -1, FEATURES_PER_PATCH, FEATURES_PER_PATCH, c_feat
                )

                weighted_features = feature_map_4d * weight_mask

                for j in range(batch.shape[0]):
                    fy, fx = coords[j]

                    feature_accumulator[
                        fy : fy + FEATURES_PER_PATCH, fx : fx + FEATURES_PER_PATCH, :
                    ] += weighted_features[j].cpu().to(torch.float32)

                    weight_accumulator[
                        fy : fy + FEATURES_PER_PATCH, fx : fx + FEATURES_PER_PATCH, :
                    ] += weight_mask.squeeze(0).cpu().to(torch.float32)
    print("Blending complete.")

    # --- F. Final Division (on CPU) ---
    print("Normalizing feature map...")
    weight_accumulator.clamp_(min=1e-6)  # Avoid division by zero
    final_seamless_map = feature_accumulator / weight_accumulator

    # Cast to float16 to save RAM
    final_seamless_map = final_seamless_map.to(torch.float16)

    # --- G. Crop Feature Map to Original Size ---
    orig_feat_h = img_h // MODEL_PATCH_SIZE
    orig_feat_w = img_w // MODEL_PATCH_SIZE
    final_feature_map = final_seamless_map[:orig_feat_h, :orig_feat_w, :]

    final_h, final_w, _ = final_feature_map.shape
    print(f"Cropped seamless map (final): {final_h}x{final_w}x{c_feat}")

    return final_feature_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DINOv3 ViT-7B blended feature extraction + PCA preview"
    )
    add_common_args(
        parser,
        default_model_path="dinov3_vit7b16.pth",
        default_output="vit_processor_test_output.png",
    )
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    model = load_vit_model(str(args.repo_dir), str(args.model_path), device)

    image = Image.open(args.image).convert("RGB")
    print(f"Image loaded successfully. Original size: {image.size[0]}x{image.size[1]}")

    feature_map = process_image_vit_blended(
        model=model,
        image=image,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    plot_pca_comparison(
        image,
        feature_map,
        model_stride=MODEL_PATCH_SIZE,
        title="DINOv3 ViT-7B Feature PCA (Blended)",
        output=args.output,
        show=not args.no_show,
    )
