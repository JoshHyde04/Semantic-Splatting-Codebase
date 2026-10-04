"""
Finds which COLMAP images see a given splat chunk, based on the chunk's XY bounding
box encoded in its filename (e.g. '5x5#-10_-10_-5_-5#-2_-2.ply' -> min_x, min_y, max_x, max_y).

Used by extract_features.py via --chunk-file / --colmap-dir.
"""

import argparse
import collections
import os
import struct
from pathlib import Path

import numpy as np

Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


# --- A. Minimal COLMAP binary reader (based on the standard COLMAP scripts) ---


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            # Skip 2D points data for speed, we only need metadata here
            fid.seek(num_points2D * (2 * 8 + 8), 1)
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=None,
                point3D_ids=None,
            )
    return images


def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            error = props[7]
            track_len = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, track_len * 2 * 4, "ii" * track_len)
            image_ids = np.array(track_elems[0::2])
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=None,
            )
    return points3D


# --- B. Chunk matching logic ---


def get_images_for_chunk(chunk_filename, colmap_path, margin: float = 1.0):
    """
    Returns a sorted list of image filenames that see any COLMAP point inside the
    chunk's XY bounds (Z ignored), expanded by `margin`.

    chunk_filename: name of the form '<prefix>#min_x_min_y_max_x_max_y#<suffix>.ply'
    colmap_path:    sparse model folder containing points3D.bin and images.bin
    """
    # 1. Parse the bounding box out of the filename
    try:
        bbox_str = Path(chunk_filename).name.split("#")[1]  # "-10_-10_-5_-5"
        min_x, min_y, max_x, max_y = (float(v) for v in bbox_str.split("_"))
    except Exception as e:
        print(f"Error parsing chunk filename '{chunk_filename}': {e}")
        return []

    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    print(f"Filtering for Chunk Bounds: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")

    # 2. Load COLMAP data
    print("Loading COLMAP data...")
    points3D = read_points3D_binary(os.path.join(colmap_path, "points3D.bin"))
    images = read_images_binary(os.path.join(colmap_path, "images.bin"))

    # 3. Any image that sees a point inside the box is relevant
    relevant_image_ids = set()
    for point in points3D.values():
        x, y, _ = point.xyz
        if min_x <= x <= max_x and min_y <= y <= max_y:
            relevant_image_ids.update(int(i) for i in point.image_ids)

    # 4. Ids -> filenames (sorted so runs are reproducible)
    return sorted(images[i].name for i in relevant_image_ids if i in images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List the COLMAP images that see a splat chunk"
    )
    parser.add_argument(
        "--chunk-file", required=True, help="e.g. '5x5#-10_-10_-5_-5#-2_-2.ply'"
    )
    parser.add_argument(
        "--colmap-dir",
        type=Path,
        required=True,
        help="Sparse model folder (contains points3D.bin and images.bin)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Extra XY margin around the chunk bounds",
    )
    args = parser.parse_args()

    names = get_images_for_chunk(args.chunk_file, str(args.colmap_dir), args.margin)
    print(f"Found {len(names)} relevant images for this chunk.")
    print("Sample:", names[:5])
