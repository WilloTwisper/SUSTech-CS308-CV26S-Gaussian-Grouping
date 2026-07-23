## Project Overview
**Computer Vision 2026 Final Project — Topic 5: Segmentation in 3D Gaussian Splatting**

Based on the [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) paper (ECCV 2024), this project involves object-level segmentation on 3DGS scenes and downstream editing tasks.

## Core Repository
- **Primary codebase**: `https://github.com/lkeab/gaussian-grouping` (Apache-2.0, Python + CUDA)
- **Working location**: `/home/jialiang_liu/gaussian-grouping` (WSL2)
- **Conda environment**: `mamba_env` (Python 3.10, PyTorch 2.10.0+cu128, CUDA 12.8)

## Project Tasks
1. **Method Reproduction**
   - Object segmentation on 3DGS scenes
   - Downstream tasks: 3D Object Removal, Inpainting, Style Transfer, Multi-Object Editing
   - Run on ≥3 scenes from LERF-Mask + Mip-NeRF360 datasets
2. **Own Contribution**
   - Custom real-world data capture & validation
   - Method enhancements (parameter tuning, improvements, etc.)

## Datasets
- **LERF-Mask** (3 scenes: `figurines`, `ramen`, `teatime`): Downloaded from HuggingFace mirror
  - Located at `data/lerf_mask/<scene>/`
  - Pre-converted with object masks in `object_mask/` directory
- **Mip-NeRF360**: https://jonbarron.info/mipnerf360/
- **Pre-trained checkpoints**: `checkpoint/lerf_mask/<scene>/` (from HuggingFace)

## Key Commands

```bash
# Access WSL2 environment
wsl bash -c 'source /home/jialiang_liu/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env && cd /home/jialiang_liu/gaussian-grouping && <command>'

# Train (memory-heavy, prefer pre-trained checkpoints)
python train.py -s data/lerf_mask/figurines -r 4 -m output/figurines --config_file config/gaussian_dataset/train.json --train_split

# Segmentation rendering (from pretrained)
python render.py -m output/figurines_pretrained -s data/lerf_mask/figurines --num_classes 256 --images images

# LERF-Mask rendering with text prompts (requires Grounded-SAM)
python render_lerf_mask.py -m output/figurines_pretrained --skip_train

# Evaluation
python script/eval_lerf_mask.py figurines

# Object removal (requires config with select_obj_id)
python edit_object_removal.py -m output/figurines_pretrained -s data/lerf_mask/figurines --config_file config/removal.json

# Object inpainting
python edit_object_inpaint.py -m output/figurines_pretrained -s data/lerf_mask/figurines --config_file config/inpaint.json
```

## Critical Fixes Applied

### 1. Blackwell GPU (RTX 5070 Ti, compute capability 12.0) — GCC 13 Strict Mode
Two CUDA submodules needed patches for GCC 13's strict C++17 compliance:
- **`submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`**: Added `#include <cstdint>` (missing `uint32_t`, `uint64_t`, `std::uintptr_t`)
- **`submodules/simple-knn/simple_knn.cu`**: Added `#include <cfloat>` (missing `FLT_MAX`)
- Compile with: `pip install -e . --no-build-isolation` (torch must be in PATH during build)
- Set `export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0"` before compilation

### 2. Object Mask Resolution Mismatch with `-r` flag
**Bug**: `utils/camera_utils.py:loadCam()` resizes RGB images to match `-r` scale but does NOT resize object masks, causing shape mismatch in training loss.
**Fix**: Added PIL resize with NEAREST interpolation for object masks in `camera_utils.py`.

### 3. Edit Scripts Save/Load Path Mismatch
**Bug**: `edit_object_removal.py` saves to `point_cloud_object_removal/` but loads from `_object_removal/`. Same for inpainting.
**Fix**: Changed save path to `point_cloud/_object_removal/iteration_<N>/` (and `_object_inpaint/`).

### 4. Empty View List Crash in render_set
**Bug**: `render_set()` in edit scripts crashes with `UnboundLocalError` when views list is empty.
**Fix**: Added `if len(views) == 0: return` guard.

## HuggingFace Mirror
- Original HuggingFace is blocked in China. Use `hf-mirror.com`:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

## Pre-trained Model Setup
```bash
mkdir -p output/<scene>_pretrained/point_cloud/iteration_30000
cp checkpoint/lerf_mask/<scene>/point_cloud/iteration_30000/* output/<scene>_pretrained/point_cloud/iteration_30000/
cp checkpoint/lerf_mask/<scene>/cfg_args output/<scene>_pretrained/
```

## Architecture Notes
- Code is forked from original 3D Gaussian Splatting; assumes Linux. WSL2 required on Windows.
- Training adds Identity Encoding (classifier on top of Gaussian features) — increases VRAM vs vanilla 3DGS
- Editing scripts create new Gaussians with removed/inpainted objects as separate point_cloud subdirectories
- Scene loading requires `source_path` (for cameras) + `model_path` (for trained Gaussians)

## Report
- Template: `Template.md` and `Report Template.pdf`
- Due: **2026.6.21** (end of day)
- Submission: PDF report + supplementary zip + code zip
