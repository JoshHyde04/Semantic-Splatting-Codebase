# DINOv3 features on Gaussian splats

Tools for extracting dense [DINOv3](https://github.com/facebookresearch/dinov3) features from large images and projecting them onto a 3D Gaussian splat, so each Gaussian carries a semantic feature vector. The splat is also coloured by a PCA of those features, which makes the semantic structure of a scene visible at a glance.

The pipeline has two stages:

```
images ──► feature_extractor.py ──► per-image feature maps (.safetensors)
                                           │
 splat .ply + COLMAP cameras + images ─────┴──► semantic_splatter.py ──► coloured .ply
                                                                      + per-Gaussian features (.npy)
```

## Contents

| File | Purpose |
|---|---|
| `feature_extractor.py` | Stage 1. Runs DINOv3 over a folder of images, compresses features with PCA, saves one `.safetensors` per image. |
| `semantic_splatter.py` | Stage 2. Projects the feature maps onto a Gaussian splat using COLMAP cameras, averages across views, colours the splat. |
| `vit_processor.py` | DINOv3 ViT-7B tiled/blended extraction for large images. Also runs standalone as a PCA preview. |
| `convnext_processor.py` | DINOv3 ConvNeXt-L single-pass extraction. Also runs standalone as a PCA preview. |
| `dinov3_utils.py` | Shared helpers: DINOv3 repo auto-clone, model loading, preprocessing, PCA plotting, feature-dim detection. |
| `chunk_filter.py` | Optional. Finds which COLMAP images see a given splat chunk. |
| `pyproject.toml` | Dependencies, managed with [uv](https://docs.astral.sh/uv/). |

## Requirements

**Hardware**
- An NVIDIA GPU is strongly recommended. The ViT-7B weights alone need roughly 14 GB of VRAM in bf16, plus activations, so a 24 GB card is comfortable. ConvNeXt-L needs far less.
- CPU-only works in principle but is impractically slow for ViT-7B.
- Several GB of system RAM: the ViT blending step accumulates a float32 feature map on the CPU (about 0.75 GB for a 4000x3000 image).
- `pyproject.toml` installs CUDA 12.8 builds of PyTorch on Windows and Linux. On macOS it falls back to the default PyPI build (CPU/MPS is not specially handled). For a different CUDA version, edit `cu128` in the `[[tool.uv.index]]` and `[tool.uv.sources]` sections.

**Software**
- [uv](https://docs.astral.sh/uv/getting-started/installation/). It creates the environment and installs everything.
- `git` on your PATH, and internet access on the first run, because the DINOv3 repo is cloned automatically into `./dinov3`. If you'd rather clone it yourself: `git clone https://github.com/facebookresearch/dinov3.git dinov3`.
- Python 3.10 to 3.13 (`uv` will fetch a suitable one).

**Python dependencies** (installed by `uv` from `pyproject.toml`): torch, torchvision, pillow, numpy, matplotlib, scikit-learn, scipy, joblib, plyfile, safetensors, tqdm, plus `omegaconf`, `termcolor` and `torchmetrics`. The last three are there because DINOv3's `hubconf.py` may import modules that need them. If model loading fails with a missing-module error, add that module to `pyproject.toml`.

**DINOv3 weights (you must obtain these yourself)**
The weights are gated. Request access via Meta's DINOv3 release, download the `.pth` files, and place them where you like (default expected names in the working directory):
- `dinov3_vit7b16.pth` for `--model vit`
- `dinov3_convnext_large.pth` for `--model convnext`

Check the DINOv3 licence terms before redistributing weights. This repo does not include them.

## Input data requirements

Stage 1 only needs a folder of images. Stage 2 additionally needs:

1. **A trained 3D Gaussian splat as a `.ply`**, in the standard 3DGS layout. Fields used: `x, y, z`, `opacity` (logit; if missing, 1.0 is assumed), `rot_0..rot_3`, `scale_0..scale_2` (log-scale, used to derive each Gaussian's normal), and `f_dc_0..f_dc_2` (the output colours are written here).
2. **A COLMAP sparse model in binary format**: `cameras.bin` and `images.bin` (and `points3D.bin` if you use `chunk_filter.py`), usually in `sparse/0`.
3. **The source images**, with names matching the COLMAP `images.bin` entries (relative to `--images-dir`).

Important assumptions:
- The splat and the COLMAP model must be in the **same coordinate frame**.
- Cameras are treated as **pinhole with no lens distortion** (only focal length and principal point are used). Use undistorted images and a pinhole/simple-pinhole COLMAP model.
- The upsampling step in `semantic_splatter.py` assumes **landscape** images (width is the longer side).
- Image names must be **unique by filename stem**. Feature files are named after the stem, so `a/IMG1.jpg` and `b/IMG1.jpg` would collide. The extractor stops with an error if this happens.

## Quick start

```bash
# 1. Extract features for all images in a folder (ViT-7B by default)
uv run feature_extractor.py --images-dir path/to/images

# 2. Project them onto a splat
uv run semantic_splatter.py \
    --ply regular.ply \
    --colmap-dir path/to/colmap/sparse/0 \
    --images-dir path/to/images \
    --features-dir features_vit
```

The first `uv run` creates a virtual environment and downloads PyTorch (a few GB). After that, runs start quickly. Quote any path that contains spaces.

## Example dataset and exact run

The pipeline was developed and tested on the **Tabuhan P1 (10 Feb 2025)** survey from the open [Sweet Corals](https://huggingface.co/datasets/wildflow/sweet-corals) coral-reef photogrammetry dataset (also at [rayhankmm/sweet-corals](https://huggingface.co/datasets/rayhankmm/sweet-corals)), released under **CC-BY-4.0** (attribution required, see [Acknowledgements](#acknowledgements)).

Of the survey patches listed in the dataset card, Tabuhan P1 is the one with everything this pipeline needs: colour-corrected images, COLMAP poses, a 3D point cloud and a 3D Gaussian Splatting model. The corrected images are undistorted pinhole images, which matches the camera assumption above. The dataset is large (the raw images for this patch alone are about 26 GB), so download only that folder, for example:

```bash
uvx --from huggingface_hub hf download rayhankmm/sweet-corals --repo-type dataset \
    --include "_indonesia_tabuhan_p1_20250210/*" \
    --local-dir datasets/_indonesia_tabuhan_p1_20250210
```

(Check the Hugging Face CLI docs if the command differs in your version. The folder layout on the dataset page is `corrected/images`, `colmap/` and `3dgs/`; adjust the paths below to match what you downloaded.)

The experiments used the splat chunk **`5x5#-10_-10_-5_-5#-2_-2.ply`**, whose filename encodes the XY bounds `min_x=-10, min_y=-10, max_x=-5, max_y=-5`. With the dataset downloaded into `DATA`:

```bash
DATA=datasets/_indonesia_tabuhan_p1_20250210/_indonesia_tabuhan_p1_20250210

# 1. Features for only the images that see the chunk
uv run feature_extractor.py \
    --images-dir $DATA/corrected/images \
    --chunk-file "5x5#-10_-10_-5_-5#-2_-2.ply" \
    --colmap-dir $DATA/colmap/sparse/0 \
    --output-dir features_vit

# 2. Project onto the splat (add --reliability for the uncertainty outputs)
uv run semantic_splatter.py \
    --ply regular.ply \
    --colmap-dir $DATA/colmap/sparse/0 \
    --images-dir $DATA/corrected/images \
    --features-dir features_vit
```

`regular.ply` is the splat file used in the original experiments. Point `--ply` at your own splat or chunk file. Use whichever folder contains `cameras.bin`, `images.bin` and `points3D.bin` for `--colmap-dir` (`colmap/sparse/0` in the copy used here). On Windows PowerShell, use `$DATA = "datasets\..."` and `$DATA\corrected\images` instead.


## Stage 1: `feature_extractor.py`

For each image it runs DINOv3 to get a dense feature map, then compresses the channel dimension with PCA and saves `[H, W, n_components]` in float16 to `<output-dir>/<image-stem>.safetensors`.

- **Models.** `--model vit` (default) is ViT-7B with 16-pixel patches (one feature per 16x16 pixels). `--model convnext` is ConvNeXt-L with stride 32 (one feature per 32x32 pixels).
- **Tiling and blending (ViT).** Large images are split into overlapping tiles (`--patch-size 512 --stride 256`), each is run through the model, and the tile features are blended back together with a pyramid-shaped weight mask so there are no visible seams. The result covers the full image at its native resolution. ConvNeXt processes the whole image in one pass.
- **PCA compression.** Raw features are 4096-dimensional (ViT-7B) or 1536-dimensional (ConvNeXt-L). The script fits a PCA on a random sample of pixels from a few random images (`--pca-samples`, `--pca-pixels`) and projects every image onto the first `--n-components` components (default 64). The fitted PCA is saved to `pca_<model>_<n>.joblib` and reused on later runs, so every image shares the same projection. Delete it, or point `--pca-path` elsewhere, if you change `--n-components` or the data.
- **Skipping.** Existing output files are skipped, so an interrupted run can be resumed.

Useful options:

| Option | Meaning |
|---|---|
| `--output-dir` | Default `features_<model>` |
| `--n-components` | Compressed channels (default 64). `--no-pca` saves raw features (very large) |
| `--batch-size` | ViT tiles per batch (default 8). Lower it if you run out of VRAM |
| `--image-list` | Text file of image paths (relative to `--images-dir`), one per line |
| `--chunk-file` + `--colmap-dir` | Only process images that see a splat chunk (uses `chunk_filter.py`) |
| `--model-path`, `--repo-dir` | Weights file and DINOv3 clone locations |
| `--seed` | Seeds the PCA image/pixel sampling and PCA solver |

Images are found recursively and the formats `.jpg .jpeg .png .tif .tiff .bmp .webp` are picked up.

## Stage 2: `semantic_splatter.py`

For every COLMAP camera:

1. Project all Gaussian centres into the image and keep those in front of the camera and inside the frame. Cameras that see fewer than 0.1% of the Gaussians are skipped.
2. Load that image's feature map and **upsample** it to a 2048-pixel-wide map, first bilinearly, then with a **guided filter** that uses the RGB image as guidance. This snaps feature edges to real image edges (for example coral against sand).
3. Run **occlusion culling** with a depth buffer, so a Gaussian hidden behind another surface doesn't pick up the wrong view's features (`--depth-threshold`).
4. Sample the feature map at each visible Gaussian's projected position.
5. Accumulate with a **confidence weight** equal to `1 / distance²  x  |cos(angle between view direction and Gaussian normal)|  x  opacity`. Closer, more face-on, more opaque Gaussians count more.

After all views, each Gaussian's feature is the confidence-weighted mean. A 3-component PCA of those features gives an RGB colour, written into the PLY's `f_dc_*` fields (and `red/green/blue` if present).

Outputs, next to `--output` (default `semantic_model.ply`):

| File | Content |
|---|---|
| `<output>.ply` | The input splat, with colours replaced by the feature PCA |
| `<output>_features.npy` | `[N, D]` feature per Gaussian, in the same order as the PLY vertices. Use this for semantic search or clustering |

### `--reliability` mode

Adds a quality check on the features:

- Tracks how much each Gaussian's feature varies across the views that see it, and converts that into a normalised uncertainty score.
- Subtracts the model-wide mean feature (global centring). The saved `_features.npy` is centred in this mode.
- Compares each Gaussian's uncertainty with its `--reliability-k` nearest neighbours (default 50). Gaussians that are much more uncertain than their surroundings are scaled down, to as little as 30%. `--reliability-alpha` sets the strength.
- The reliability weighting is only applied to the PCA used for the colours. Nothing is removed from the model, and the saved `_features.npy` is not weighted.

Extra outputs: `<output>_view_counts.npy` (views per Gaussian), `<output>_cv.npy` (feature spread relative to magnitude) and `<output>_uncertainty.npy` (normalised uncertainty). These are meant for validating feature quality.

Other options: `--feature-dim` (auto-detected from the feature files), `--guided-filter-radius`, `--guided-filter-eps`, and `--knn-k` (optional colour smoothing over spatial neighbours, off by default).

## Standalone previews

`vit_processor.py` and `convnext_processor.py` run on a single image and save a side-by-side of the image and its feature PCA. This is a quick way to check the model works and see what features look like:

```bash
uv run vit_processor.py --image path/to/image.jpg --model-path dinov3_vit7b16.pth
uv run convnext_processor.py --image path/to/image.jpg --model-path dinov3_convnext_large.pth
```

## Working on chunks of a large scene

If your splat is split into chunks whose filenames encode the XY bounds (`<prefix>#min_x_min_y_max_x_max_y#<suffix>.ply`, for example `5x5#-10_-10_-5_-5#-2_-2.ply`), you can extract features only for images that see a chunk:

```bash
uv run feature_extractor.py --images-dir path/to/images \
    --chunk-file "5x5#-10_-10_-5_-5#-2_-2.ply" \
    --colmap-dir path/to/colmap/sparse/0
```

`chunk_filter.py` reads `points3D.bin`, finds COLMAP points whose XY fall inside the chunk bounds (plus a 1.0 margin, Z ignored) and returns every image that observes any of them. You can also run it directly to list the images.

## Reproducing results

- Commit the generated **`uv.lock`** so everyone gets identical package versions.
- Use the same `--seed` (default 0), images, and PCA settings. The PCA file can also be shared to guarantee an identical projection.
- The DINOv3 repo is cloned from its default branch, so for exact reproduction check out the same commit inside `./dinov3`.
- GPU and driver differences can cause tiny numerical differences in bf16 inference, so expect near-identical rather than bit-identical results.

## Troubleshooting

- **Out of GPU memory (ViT):** lower `--batch-size`, or reduce `--patch-size`.
- **`Weights not found`:** download the gated `.pth` file and pass `--model-path`.
- **Missing module when loading the model:** add it to `pyproject.toml` (DINOv3's hub code can pull in extra imports).
- **Could not clone DINOv3:** install `git`, or clone manually into `./dinov3`.
- **Feature-dimension errors in stage 2:** `--feature-dim` is auto-detected, so this usually means the features folder mixes outputs from different runs. Use a clean `--output-dir`.
- **`No .safetensors feature files found`:** check `--features-dir` points at stage 1's output folder.
- **Few or no images matched in stage 2:** check that the COLMAP image names match files under `--images-dir`, and that the splat and COLMAP model share a coordinate frame.

## Known limitations

- Pinhole cameras only, with distortion ignored.
- Stage 2 assumes landscape images.
- The ViT feature map is cropped to a multiple of 16 pixels (and ConvNeXt to a multiple of 32), so up to a few pixels at the right and bottom edges have no feature. This is negligible after upsampling.
- The chunk filter relies on the filename convention above and an XY-only bounding box.

## Acknowledgements

Example data: Sweet Corals, a coral-reef photogrammetry dataset by Wildflow, the University of Derby and BRIN (Indonesia), licensed CC-BY-4.0. If you use it, cite:

> Nozdrenkov, S., Mujiyanto, Zedta, R. R., Fakhrurrozi, Barker, T., Rahman, M. D., Samusamu, A. S., Rochman, F., Peters, L., Rachmawati, R., Johan, O., Craggs, J., Sweet, M. (2025). *sweet-corals* [Dataset]. Hugging Face. https://huggingface.co/datasets/wildflow/sweet-corals, doi: [10.57967/hf/5162](http://doi.org/10.57967/hf/5162)

Features are extracted with [DINOv3](https://github.com/facebookresearch/dinov3) by Meta AI; please follow its licence and citation requirements.
