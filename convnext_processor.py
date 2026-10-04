import argparse

import torch
from PIL import Image

from dinov3_utils import (
    add_common_args,
    build_transform,
    get_device,
    load_hub_model,
    plot_pca_comparison,
)

MODEL_STRIDE = 32  # Fixed for ConvNeXt


def load_convnext_model(repo_dir: str, model_path: str, device: torch.device):
    """Loads the DINOv3 ConvNeXT-L model to the specified device."""
    return load_hub_model("dinov3_convnext_large", repo_dir, model_path, device)


def process_image_convnext_singlepass(
    model: torch.nn.Module, image: Image.Image, device: torch.device
):
    """
    Processes a large PIL Image in a single pass (whole-image-at-once).
    This is possible because the ConvNeXT-L and its activations fit
    in VRAM, as proven by testing.

    Returns:
        torch.Tensor: The final feature map, on the CPU in float16.
    """
    print(f"Starting single-pass processing for image of size {image.size}...")

    img_tensor = build_transform()(image).unsqueeze(0).to(device)
    _, _, h_in, w_in = img_tensor.shape
    print(f"Full-resolution tensor created on GPU: {img_tensor.shape}")

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        features_dict = model.forward_features(img_tensor)
        # feature_map is [1, N, C]
        feature_map_patches = features_dict["x_norm_patchtokens"]

    print(f"Inference complete. Output patch shape: {feature_map_patches.shape}")

    h_feat = h_in // MODEL_STRIDE
    w_feat = w_in // MODEL_STRIDE
    _b, n, c_from_model = feature_map_patches.shape

    if n != (h_feat * w_feat):
        print(f"WARNING: Patch count mismatch. {n} != {h_feat * w_feat}")

    final_feature_map = feature_map_patches.reshape(h_feat, w_feat, c_from_model)
    print(f"Reshaped to final map: {final_feature_map.shape}")

    return final_feature_map.cpu().to(torch.float16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DINOv3 ConvNeXt-L single-pass feature extraction + PCA preview"
    )
    add_common_args(
        parser,
        default_model_path="dinov3_convnext_large.pth",
        default_output="convnext_processor_test_output_singlepass.png",
    )
    args = parser.parse_args()

    device = get_device()
    model = load_convnext_model(str(args.repo_dir), str(args.model_path), device)

    image = Image.open(args.image).convert("RGB")
    print(f"Image loaded successfully. Original size: {image.size[0]}x{image.size[1]}")

    feature_map = process_image_convnext_singlepass(model, image, device)

    plot_pca_comparison(
        image,
        feature_map,
        model_stride=MODEL_STRIDE,
        title="DINOv3 ConvNeXt-L PCA (Single Pass)",
        output=args.output,
        show=not args.no_show,
    )
