"""
Projects per-image DINOv3 feature maps (from extract_features.py) onto a Gaussian-splat
PLY using the COLMAP cameras, averages them per Gaussian with confidence weighting
(distance, viewing angle, opacity), and colours the splat by a 3-component PCA.

With --reliability it additionally estimates per-Gaussian feature uncertainty across
views, centres the features, and down-weights locally unreliable Gaussians before the
PCA colouring. It also saves view-count / CV / uncertainty arrays for validation.
"""

import argparse
import os
import struct
from pathlib import Path

import numpy as np
import scipy.spatial
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from plyfile import PlyData, PlyElement
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from tqdm import tqdm

from dinov3_utils import get_device, infer_feature_dim

DEVICE = get_device()

# Tunable Parameters (defaults)
DEPTH_THRESHOLD = 0.1  # Occlusion culling threshold
GRAZING_ANGLE_THRESH = 0.2  # Angle culling threshold
KNN_K = 15  # Number of neighbors for smoothing
GUIDED_FILTER_RADIUS = 2  # Radius for upsampling filter (2-4 is good)
GUIDED_FILTER_EPS = 1e-2  # Smoothness regularization


# --- 1. Guided Filter (Upsampling) ---
def fast_guided_filter(guidance, inputs, radius, eps):
    """
    Functional implementation optimized for VRAM.
    Uses Grayscale guidance to align features.
    guidance: [1, 3, H, W] (RGB)
    inputs: [1, C, H, W] (Features)
    """
    # Grayscale guidance captures the luminance edges (coral vs sand)
    guide = guidance.mean(dim=1, keepdim=True)  # [1, 1, H, W]

    box = torch.nn.AvgPool2d(kernel_size=2 * radius + 1, stride=1, padding=0)

    def box_filter(x):
        x_padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        return box(x_padded)

    mean_I = box_filter(guide)
    mean_p = box_filter(inputs)
    mean_Ip = box_filter(guide * inputs)  # [1, C, H, W]
    mean_II = box_filter(guide * guide)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a)
    mean_b = box_filter(b)

    q = mean_a * guide + mean_b
    return q


# --- 2. Helper Math Functions ---
def build_rotation(r):
    norm = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)
    q = r / norm[:, None]
    R = torch.zeros((q.shape[0], 3, 3), device=DEVICE)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def compute_gaussian_normals(plydata):
    print("Deriving Gaussian normals...")
    v = plydata["vertex"]
    rots = torch.stack(
        [
            torch.tensor(v["rot_0"], device=DEVICE),
            torch.tensor(v["rot_1"], device=DEVICE),
            torch.tensor(v["rot_2"], device=DEVICE),
            torch.tensor(v["rot_3"], device=DEVICE),
        ],
        dim=1,
    ).float()
    scales = torch.stack(
        [
            torch.tensor(v["scale_0"], device=DEVICE),
            torch.tensor(v["scale_1"], device=DEVICE),
            torch.tensor(v["scale_2"], device=DEVICE),
        ],
        dim=1,
    ).float()

    R = build_rotation(rots)
    min_scale_idx = torch.argmin(scales, dim=1)
    normals = torch.gather(
        R, 2, min_scale_idx.unsqueeze(1).unsqueeze(2).expand(-1, 3, 1)
    ).squeeze(2)
    return normals


def perform_occlusion_culling(
    uv_coords, depth_values, img_w, img_h, threshold=DEPTH_THRESHOLD
):
    px = ((uv_coords[:, 0] + 1) / 2 * img_w).long().clamp(0, img_w - 1)
    py = ((uv_coords[:, 1] + 1) / 2 * img_h).long().clamp(0, img_h - 1)
    linear_idx = py * img_w + px
    num_pixels = img_w * img_h
    z_buffer = torch.full((num_pixels,), float("inf"), device=DEVICE)
    z_buffer.scatter_reduce_(
        0, linear_idx, depth_values, reduce="min", include_self=False
    )
    nearest_depths = z_buffer[linear_idx]
    is_visible = depth_values <= (nearest_depths + threshold)
    return is_visible


# --- 3. Loaders ---
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            if model_id == 0:
                num_params = 3
            elif model_id == 1:
                num_params = 4
            elif model_id == 2:
                num_params = 4
            elif model_id == 3:
                num_params = 5
            elif model_id == 4:
                num_params = 8
            else:
                num_params = 4
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = {
                "w": camera_properties[2],
                "h": camera_properties[3],
                "params": np.array(params),
            }
    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2D * 24, 1)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
            }
    return images


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def load_ply_data_enhanced(path):
    print(f"Loading PLY: {path}")
    plydata = PlyData.read(path)
    x = torch.tensor(plydata["vertex"]["x"], device=DEVICE, dtype=torch.float32)
    y = torch.tensor(plydata["vertex"]["y"], device=DEVICE, dtype=torch.float32)
    z = torch.tensor(plydata["vertex"]["z"], device=DEVICE, dtype=torch.float32)
    xyz = torch.stack([x, y, z], dim=1)
    if "opacity" in plydata["vertex"].data.dtype.names:
        op_logits = torch.tensor(
            plydata["vertex"]["opacity"], device=DEVICE, dtype=torch.float32
        )
        opacity = torch.sigmoid(op_logits)
    else:
        print("[Warning] 'opacity' not found. Defaulting to 1.0")
        opacity = torch.ones_like(x)
    return xyz, opacity, plydata


def save_semantic_ply(original_plydata, semantic_colors, output_path):
    print(f"Saving Semantic PLY to {output_path}...")
    C0 = 0.28209479177387814
    rgb = semantic_colors.clip(0, 1)
    sh_dc = (rgb - 0.5) / C0
    vertex = original_plydata["vertex"].data.copy()
    vertex["f_dc_0"] = sh_dc[:, 0]
    vertex["f_dc_1"] = sh_dc[:, 1]
    vertex["f_dc_2"] = sh_dc[:, 2]
    if "red" in vertex.dtype.names:
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        vertex["red"] = rgb_uint8[:, 0]
        vertex["green"] = rgb_uint8[:, 1]
        vertex["blue"] = rgb_uint8[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(output_path)
    print("Save Complete.")


def get_projection_matrix(w, h, params):
    if len(params) == 3:
        fx, fy, cx, cy = params[0], params[0], params[1], params[2]
    else:
        fx, fy, cx, cy = params[:4]
    return torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=DEVICE, dtype=torch.float32
    )


def apply_knn_smoothing(rgb_norm, xyz, k=10):
    print(f"Applying KNN Smoothing (k={k})...")
    if isinstance(xyz, torch.Tensor):
        points_np = xyz.cpu().numpy()
    else:
        points_np = xyz

    colors_np = rgb_norm
    tree = scipy.spatial.cKDTree(points_np)
    smoothed_colors = np.zeros_like(colors_np)
    batch_size = 10000
    n_points = points_np.shape[0]

    for i in tqdm(range(0, n_points, batch_size), desc="Smoothing"):
        end = min(i + batch_size, n_points)
        batch_points = points_np[i:end]
        dists, idxs = tree.query(batch_points, k=k, workers=-1)
        neighbor_colors = colors_np[idxs]
        mean_col = neighbor_colors.mean(axis=1)
        smoothed_colors[i:end] = mean_col
    return smoothed_colors


# --- 4. Semantic reliability (used with --reliability) ---
def compute_local_reliability(xyz, u_norm, k=50, alpha=1.0):
    """
    Compute soft semantic reliability weight from local z-score.
    Returns reliability in (0,1].
    """
    xyz_np = xyz.cpu().numpy()
    tree = scipy.spatial.cKDTree(xyz_np)
    N = xyz.shape[0]

    reliability = torch.ones(N, device=xyz.device)

    batch_size = 50000
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xyz_chunk = xyz_np[start:end]

        _, idxs = tree.query(xyz_chunk, k=k + 1)
        idxs = idxs[:, 1:]

        neighbor_uncert = u_norm[idxs].cpu().numpy()
        local_mean = neighbor_uncert.mean(axis=1)
        local_std = neighbor_uncert.std(axis=1) + 1e-6

        local_z = (u_norm[start:end].cpu().numpy() - local_mean) / local_std

        # Only penalize positive outliers
        local_z = np.clip(local_z, 0.0, None)

        rel = np.exp(-alpha * local_z)
        rel = np.clip(rel, 0.3, 1.0)

        reliability[start:end] = torch.tensor(rel, device=xyz.device)

        print(f"Processed {end}/{N}")

    print(
        "Reliability stats:",
        reliability.mean().item(),
        reliability.min().item(),
        reliability.max().item(),
    )

    return reliability


# --- OPTIMIZED PROCESSING LOOP ---
def process_semantic_colorization(
    ply_model_path: str,
    features_dir: str,
    colmap_dir: str,
    images_dir: str,
    output_ply_path: str,
    feature_dim: int | None = None,
    depth_threshold: float = DEPTH_THRESHOLD,
    guided_filter_radius: int = GUIDED_FILTER_RADIUS,
    guided_filter_eps: float = GUIDED_FILTER_EPS,
    knn_k: int = 0,
    reliability: bool = False,
    reliability_k: int = 50,
    reliability_alpha: float = 1.0,
):
    if feature_dim is None:
        feature_dim = infer_feature_dim(features_dir)

    print("Loading COLMAP binary files...")
    cameras = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(colmap_dir, "images.bin"))

    # Load Geometry
    xyz_world, opacity, plydata = load_ply_data_enhanced(ply_model_path)
    normals = compute_gaussian_normals(plydata)

    n_points = xyz_world.shape[0]
    feature_accum = torch.zeros(
        (n_points, feature_dim), device=DEVICE, dtype=torch.float32
    )
    weight_accum = torch.zeros((n_points, 1), device=DEVICE, dtype=torch.float32)
    if reliability:
        feature_sq_accum = torch.zeros_like(feature_accum)
        view_count = torch.zeros((n_points, 1), device=DEVICE)

    print(f"Processing {len(images)} images (Smart Culling + 2048px Limit)...")
    matched_images = 0
    skipped_images = 0

    transform = T.Compose([T.ToTensor()])

    with torch.no_grad():
        for img_id, img_data in tqdm(images.items()):
            # --- 1. GEOMETRY CHECK (Do this FIRST) ---
            R_cam = torch.tensor(
                qvec2rotmat(img_data["qvec"]), device=DEVICE, dtype=torch.float32
            )
            t_cam = torch.tensor(img_data["tvec"], device=DEVICE, dtype=torch.float32)
            cam = cameras[img_data["camera_id"]]

            xyz_cam = torch.matmul(xyz_world, R_cam.T) + t_cam
            mask_z = xyz_cam[:, 2] > 0.1  # In front of camera

            if not mask_z.any():
                skipped_images += 1
                continue

            K = get_projection_matrix(cam["w"], cam["h"], cam["params"])
            uv = torch.matmul(xyz_cam, K.T)
            uv = uv[:, :2] / uv[:, 2:3]

            # uv is in pixels here, before the grid normalisation below
            u_px, v_px = uv[:, 0], uv[:, 1]
            in_view_mask = (
                (u_px > 0) & (u_px < cam["w"]) & (v_px > 0) & (v_px < cam["h"]) & mask_z
            )

            # If fewer than 0.1% of points are visible, skip this image
            if in_view_mask.sum() < (n_points * 0.001):
                skipped_images += 1
                continue

            # --- 2. NOW LOAD DATA (Only for relevant images) ---
            full_name = img_data["name"]
            base_name = os.path.basename(full_name)
            feature_name = os.path.splitext(base_name)[0] + ".safetensors"
            feature_path = os.path.join(features_dir, feature_name)
            image_path = os.path.join(images_dir, full_name)

            if not os.path.exists(feature_path):
                continue

            try:
                pil_img = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            matched_images += 1
            cam_center = -torch.matmul(R_cam.T, t_cam)

            # --- 3. VRAM SAFE RESIZING (max dimension 2048) ---
            target_w = 2048
            scale_factor = target_w / max(cam["w"], cam["h"], 1)
            new_w = int(cam["w"] * scale_factor)
            new_h = int(cam["h"] * scale_factor)

            img_resized = pil_img.resize((new_w, new_h))
            rgb_high = transform(img_resized).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

            f_tensors = load_file(feature_path)
            feats_low = list(f_tensors.values())[0].to(DEVICE).float()
            feats_low = feats_low.permute(2, 0, 1).unsqueeze(0)

            feats_bilinear = F.interpolate(
                feats_low,
                size=(new_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

            feats_refined = fast_guided_filter(
                guidance=rgb_high,
                inputs=feats_bilinear,
                radius=guided_filter_radius,
                eps=guided_filter_eps,
            )

            # --- 4. PROJECTION (Using Normalized Coordinates) ---
            # grid_sample uses [-1, 1], so it doesn't care that we resized the map
            valid_mask = in_view_mask

            xyz_cam_sub = xyz_cam[valid_mask]

            u_norm = (uv[valid_mask, 0] / cam["w"]) * 2.0 - 1.0
            v_norm = (uv[valid_mask, 1] / cam["h"]) * 2.0 - 1.0
            uv_sub = torch.stack((u_norm, v_norm), dim=-1)

            depths_sub = xyz_cam_sub[:, 2]
            is_visible_sub = perform_occlusion_culling(
                uv_sub, depths_sub, cam["w"], cam["h"], threshold=depth_threshold
            )

            if not is_visible_sub.any():
                continue

            visible_subset_indices = torch.nonzero(is_visible_sub).squeeze()
            uv_final = uv_sub[visible_subset_indices]

            global_indices = torch.nonzero(valid_mask).squeeze()
            if global_indices.dim() == 0:
                global_indices = global_indices.unsqueeze(0)
            visible_global_indices = global_indices[visible_subset_indices]

            grid = uv_final.view(1, 1, -1, 2)
            sampled = (
                F.grid_sample(
                    feats_refined,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                .flatten(2)
                .permute(2, 0, 1)
                .squeeze()
            )

            if sampled.dim() == 1:
                sampled = sampled.unsqueeze(0)

            # --- 5. WEIGHTING ---
            xyz_world_sub = xyz_world[visible_global_indices]

            dist_sq = torch.sum(xyz_cam_sub[visible_subset_indices] ** 2, dim=-1)
            w_dist = 1.0 / (dist_sq + 1e-6)

            view_dirs = cam_center - xyz_world_sub
            v_dirs = F.normalize(view_dirs, dim=1, eps=1e-6)
            norms = normals[visible_global_indices]
            w_angle = torch.abs(torch.sum(v_dirs * norms, dim=1))

            w_opacity = opacity[visible_global_indices]

            confidence = (w_dist * w_opacity * w_angle).unsqueeze(-1)

            feature_accum[visible_global_indices] += sampled * confidence
            weight_accum[visible_global_indices] += confidence
            if reliability:
                feature_sq_accum[visible_global_indices] += (sampled**2) * confidence
                view_count[visible_global_indices] += 1

            del feats_refined, feats_bilinear, rgb_high, img_resized

    print(
        f"--- STATUS: Processed {matched_images} images (Skipped {skipped_images} irrelevant views) ---"
    )
    if matched_images == 0:
        return

    weight_accum[weight_accum == 0] = 1.0
    mean_features = feature_accum / weight_accum

    pca_input = mean_features
    if reliability:
        # Per-Gaussian variance across views -> uncertainty
        mean_sq = feature_sq_accum / weight_accum
        variance = torch.clamp(mean_sq - mean_features**2, min=0.0)
        semantic_uncertainty = variance.mean(dim=1)

        # Arrays saved for validation
        sigma = torch.sqrt(variance.mean(dim=1))
        mu_norm = torch.norm(mean_features, dim=1)
        np.save(
            output_ply_path.replace(".ply", "_view_counts.npy"),
            view_count.squeeze().cpu().numpy(),
        )
        np.save(
            output_ply_path.replace(".ply", "_cv.npy"),
            (sigma / (mu_norm + 1e-8)).cpu().numpy(),
        )

        # Global feature centering over the whole 3D model
        print("Applying global feature centering to 3D bank...")
        mean_features -= mean_features.mean(dim=0, keepdim=True)

        u = semantic_uncertainty
        u_norm = (u - u.min()) / (u.max() - u.min() + 1e-6)

        # Soft per-Gaussian reliability from the local uncertainty z-score
        semantic_reliability = compute_local_reliability(
            xyz_world, u_norm, k=reliability_k, alpha=reliability_alpha
        )
        pca_input = mean_features * semantic_reliability.unsqueeze(1)
        np.save(
            output_ply_path.replace(".ply", "_uncertainty.npy"), u_norm.cpu().numpy()
        )

    full_features_path = output_ply_path.replace(".ply", "_features.npy")
    print(
        f"Saving full {mean_features.shape[1]}-dim features to {full_features_path}..."
    )
    np.save(full_features_path, mean_features.cpu().numpy())

    print("Running PCA...")
    pca = PCA(n_components=3)
    rgb_pca = pca.fit_transform(pca_input.cpu().numpy())
    rgb_norm = (rgb_pca - rgb_pca.min(0)) / (rgb_pca.max(0) - rgb_pca.min(0) + 1e-6)

    if knn_k > 0:
        rgb_norm = apply_knn_smoothing(rgb_norm, xyz_world, k=knn_k)

    save_semantic_ply(plydata, rgb_norm, output_ply_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project 2D DINOv3 features onto a Gaussian splat PLY and colour it by feature PCA"
    )
    parser.add_argument(
        "--ply", type=Path, default=Path("regular.ply"), help="Input Gaussian splat PLY"
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("features_vit"),
        help="Folder of per-image .safetensors feature maps",
    )
    parser.add_argument(
        "--colmap-dir",
        type=Path,
        required=True,
        help="COLMAP sparse model folder (contains cameras.bin and images.bin)",
    )
    parser.add_argument(
        "--images-dir", type=Path, required=True, help="Folder of source images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("semantic_model.ply"),
        help="Output PLY (.npy files are saved alongside)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=None,
        help="Channel count of the feature maps (default: auto-detect)",
    )
    parser.add_argument("--depth-threshold", type=float, default=DEPTH_THRESHOLD)
    parser.add_argument(
        "--guided-filter-radius", type=int, default=GUIDED_FILTER_RADIUS
    )
    parser.add_argument("--guided-filter-eps", type=float, default=GUIDED_FILTER_EPS)
    parser.add_argument(
        "--knn-k",
        type=int,
        default=0,
        help="KNN colour smoothing neighbours (0 = off, as in the original)",
    )
    parser.add_argument(
        "--reliability",
        action="store_true",
        help="Estimate per-Gaussian uncertainty, centre features and down-weight unreliable Gaussians; saves extra validation .npy files",
    )
    parser.add_argument(
        "--reliability-k",
        type=int,
        default=50,
        help="Neighbours for the local reliability z-score",
    )
    parser.add_argument(
        "--reliability-alpha",
        type=float,
        default=1.0,
        help="Strength of the reliability penalty",
    )
    args = parser.parse_args()

    process_semantic_colorization(
        ply_model_path=str(args.ply),
        features_dir=str(args.features_dir),
        colmap_dir=str(args.colmap_dir),
        images_dir=str(args.images_dir),
        output_ply_path=str(args.output),
        feature_dim=args.feature_dim,
        depth_threshold=args.depth_threshold,
        guided_filter_radius=args.guided_filter_radius,
        guided_filter_eps=args.guided_filter_eps,
        knn_k=args.knn_k,
        reliability=args.reliability,
        reliability_k=args.reliability_k,
        reliability_alpha=args.reliability_alpha,
    )
