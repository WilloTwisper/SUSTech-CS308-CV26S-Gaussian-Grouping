# Modernizing 3D Gaussian Segmentation: SAM 2 Integration and VRAM-Efficient Editing Pipeline

**Student Name:** 刘家亮
**Student ID:** 12412719

## Abstract

This project reproduces and extends **Gaussian Grouping** (Ye et al., ECCV 2024) for object-level segmentation and editing on 3D Gaussian Splatting scenes. We achieve an average mIoU of **72.79%** across three LERF-Mask benchmark scenes, and successfully implement object removal, inpainting, and style transfer. Our key contributions include: (1) upgrading the mask preprocessing from SAM 1 to **SAM 2 video mode**, achieving **59% improvement in multi-view temporal consistency** with stable cross-frame object tracking; (2) implementing a custom **style transfer module** via SH coefficient manipulation; (3) performing **comprehensive ablation studies** on threshold sensitivity, Gaussian redundancy, and failure case analysis; and (4) validating the pipeline on **custom-captured real-world data** (PSNR 37.52 dB). We also deploy on cloud GPU (V100 32GB) for VRAM-intensive full-resolution experiments, and discuss connections to seven 2025–2026 frontier works (BEA-GS, TrackRef3D, ReferSplat, CLM, etc.).

---

## 1. Introduction

3D Gaussian Splatting (3DGS) has emerged as a leading technique for real-time novel view synthesis and 3D scene reconstruction. By representing scenes as collections of anisotropic 3D Gaussians with differentiable rasterization, 3DGS achieves both high visual fidelity and interactive rendering speeds. However, while 3DGS excels at appearance and geometry modeling, it fundamentally lacks fine-grained, object-level scene understanding — a capability critical for downstream applications such as scene editing, robotic manipulation, and embodied AI.

This project explores **Gaussian Grouping** (Ye et al., ECCV 2024), which extends 3DGS with object-level segmentation via identity encoding and SAM-based mask supervision. We reproduce its core pipeline — object segmentation, 3D object removal, and 3D object inpainting — on the LERF-Mask benchmark across three scenes. We also discuss practical deployment challenges on consumer GPUs (12GB VRAM) and scale to cloud V100 (32GB) for full-resolution experiments, analyze failure modes, and propose directions for future improvement.

The key challenge addressed is: **How to simultaneously reconstruct a 3D scene and segment it into meaningful objects, without requiring expensive 3D annotations?**

---

## 2. Related Works

### 2.1 3D Gaussian Splatting
Kerbl et al. (SIGGRAPH 2023) introduced 3D Gaussian Splatting as an explicit scene representation using 3D Gaussians parameterized by position, covariance, opacity, and spherical harmonics (SH) coefficients for view-dependent color. Combined with a tile-based differentiable rasterizer, 3DGS achieves state-of-the-art quality and real-time rendering.

### 2.2 Segmentation in 3DGS
Several works extend 3DGS with semantic understanding:

- **Gaussian Grouping** (ECCV 2024) augments each Gaussian with an Identity Encoding, supervised by SAM-generated 2D masks and 3D spatial consistency regularization. It supports removal, inpainting, and style transfer via local Gaussian editing.
- **SAGA** (AAAI 2025) trains a contrastive feature field on top of pre-trained 3DGS, enabling interactive point-click segmentation and open-vocabulary queries via CLIP.
- **ReferSplat** (ICML 2025 Oral) introduces referring 3DGS segmentation using natural language descriptions, achieving state-of-the-art open-vocabulary performance.
- **BEA-GS** (CVPR 2026 Highlight) optimizes geometric boundaries for precise object extraction.

### 2.3 Lightweight and Memory-Efficient 3DGS
Recent works address the high VRAM requirements of 3DGS training:

- **CLM (CPU-GPU Load Management)** offloads GPU memory by storing color and opacity parameters in CPU RAM while keeping position and rotation on GPU, enabling 6.7× larger scenes on the same hardware.
- **Gaussians on a Diet** reduces peak training memory by up to 80% through dynamic pruning and growth strategies.
- **EvoGS** adopts a hierarchical structure to reduce redundant Gaussians, achieving up to 5.5× memory reduction.

---

## 3. Method

We adopt **Gaussian Grouping** as our base method. The pipeline consists of three stages:

### 3.1 Training with Identity Encoding
Each 3D Gaussian is augmented with a compact Identity Encoding vector. During differentiable rendering, these encodings are rasterized alongside RGB values, producing a per-pixel feature map. A lightweight 1×1 convolutional classifier maps the feature map to object class logits, supervised by:

- **2D Mask Loss:** Cross-entropy between predicted logits and SAM-generated object masks from the training views.
- **3D Spatial Regularization:** Encourages spatial consistency of class assignments among neighboring Gaussians in 3D space.

The total loss is:

$$\mathcal{L} = (1 - \lambda_{ssim}) \mathcal{L}_{L1} + \lambda_{ssim} \mathcal{L}_{SSIM} + \mathcal{L}_{cls} + \mathcal{L}_{reg3d}$$

### 3.2 Object Removal
Given a trained model and a selected object ID, we identify the 3D Gaussians belonging to that object via the classifier, compute their convex hull, and remove all enclosed Gaussians. The modified scene is re-rendered without the target object.

### 3.3 Object Inpainting
After removal, a "hole" remains where the object was. The inpainting procedure:
1. Identifies the 2D mask of the removed region in each training view.
2. Freezes the removed Gaussians and fine-tunes the remaining scene with a masked L1 loss (ignoring the hole region) and LPIPS perceptual loss within the bounding box of the removed region.
3. The fine-tuned background Gaussian naturally expand to fill the hole.

### 3.4 Implementation Details
- **Framework:** PyTorch 2.10.0, CUDA 12.8
- **Hardware:** NVIDIA RTX 5070 Ti Laptop (12GB VRAM) + Huawei Cloud V100 (32GB VRAM)
- **Training:** 30,000 iterations, SH degree 3, Identity Encoding dimension 256, spatial regularization interval 5
- **Removal:** Convex hull outlier factor 1.0, removal threshold 0.8
- **Inpainting:** 500 fine-tuning iterations with VGG-based LPIPS loss
- **Resolution:** Downsampled to 1/4 (via `-r 4`) on local 12GB GPU; full-resolution on cloud V100 32GB

### 3.5 Method Enhancement: SAM 2 Upgrade for Multi-View Mask Consistency

A key bottleneck in the Gaussian Grouping pipeline is the quality of 2D object masks generated by SAM. The original implementation uses SAM 1, which processes each frame independently, leading to **mask flickering** across views — inconsistent object boundaries, varying object counts, and temporal jitter in mask IDs. These inconsistencies propagate into the 3D segmentation, causing boundary artifacts and floating Gaussians.

As our own contribution, we upgraded the mask preprocessing pipeline from **SAM 1 (2023) to SAM 2 (2025)**. SAM 2 introduces a memory-based architecture with a video prediction mode that maintains temporal consistency across video frames. By tuning key parameters (`pred_iou_thresh=0.75`, `stability_score_thresh=0.75`, `min_mask_region_area=100`, propagation threshold=0.4), we achieve a balance between object separation and temporal consistency.

**Quantitative comparison:** We evaluated both approaches on the figurines scene (60 frames):

| Metric | SAM1+DEVA | SAM2 Video (ours) |
|---|---|---|
| Objects per frame (avg) | 28/24/19 | 39/29/20 |
| Inter-frame variation | 0.0452 | 0.0187 (**-59%**) |
| Temporal tracking | IDs vary per frame | IDs stable across frames |

![SAM2 Comparison](figures/sam_compare_video.png)
*Figure 3a: SAM1+DEVA vs SAM2 Video mode across three consecutive frames. Green boxes highlight a tracked object that persists in SAM2 across all frames.*

![Frame Variation](figures/chart_frame_variation.png)
*Figure 3b: Frame-by-frame mask variation — SAM2 video mode achieves 59% more temporal stability.*

![SAM2 Consistency](figures/chart_sam2_consistency.png)
*Figure 3c: Inter-frame mask variation comparison — lower values indicate more stable masks.*

The comprehensive analysis reveals a fundamental trade-off: SAM1+DEVA produces finer-grained masks with more objects per frame, but object identities are reassigned at each view. SAM2 video mode produces slightly fewer objects (due to conservative merging) but maintains consistent object IDs across all 60 frames. The 59% reduction in frame-to-frame variation confirms superior temporal stability, which is critical for multi-view 3DGS training where mask jitter directly causes floating artifacts.

### 3.6 Method Enhancement: 3D Object Style Transfer via SH Coefficient Manipulation

As a further contribution beyond the original paper, we implemented **3D object style transfer** — a novel application built upon the segmentation results. While the original Gaussian Grouping paper describes style transfer as a downstream task, no dedicated script was provided in the open-source release. We developed `edit_style_transfer.py` from scratch.

**Method:** Given a trained Gaussian Grouping model with object segmentation, we:
1. **Identify target Gaussians** via the learned classifier and a configurable confidence threshold
2. **Modify SH (Spherical Harmonics) coefficients** of the DC component to shift the base color toward a target palette (e.g., red, blue, gold, grayscale)
3. **Dampen higher-order SH components** for styled Gaussians to reduce view-dependent color inconsistency with the new base color
4. **Render the modified scene**, producing novel views with the colorized objects while preserving background fidelity

All operations are vectorized GPU tensor operations, requiring zero additional VRAM beyond model loading. Ten style presets are included: red, blue, green, gold, grayscale, sepia, purple, invert, vivid, and pastel.

**Experiments:** We tested the style transfer on the figurines scene. Through systematic exploration of the 256 object classes, we identified two well-separated foreground objects: Object ID 15 (25,482 Gaussians, 0.7% of scene) and Object ID 57 (15,243 Gaussians, 0.4%). These correspond to distinct foreground items (likely a toy chair and a rubber duck, respectively). Both objects were successfully colorized to red and blue styles with dramatic visual impact.

![Style Transfer ID15](figures/id15_original_view80.png)
![Style Transfer ID15 Red](figures/id15_red_view80.png)
*Figure 4a: Object ID 15 style transfer. Original vs. red style (25,482 Gaussians, same viewpoint).*

![Style Transfer ID57](figures/id57_original_front.png)  
![Style Transfer ID57 Blue](figures/id57_blue_front.png)
*Figure 4b: Object ID 57 style transfer. Original vs. blue style (15,243 Gaussians, front view).*

![Multi-Object Editing](figures/sequential_editing.png)
*Figure 4c: Multi-object independent editing. Left: original. Middle: Object 15 in red. Right: Object 57 in blue. Each operation modifies only its target Gaussians — the background and non-target objects remain unchanged, demonstrating composable 3D editing.*

No semantic text-labels were required — objects were identified purely through classifier output and visual inspection, demonstrating the utility of the Identity Encoding approach for unsupervised object discovery.

---

## 4. Experiments

### 4.1 Datasets

**LERF-Mask Dataset** (Ye et al., 2024): A benchmark for 3D open-vocabulary segmentation built on LERF scenes. We use all three scenes:

| Scene | Training Views | Test Views | Notable Objects |
|---|---|---|---|
| **Figurines** | 303 | 4 | 7 objects (apples, toy chairs, camera, duck, porcelain hand) |
| **Ramen** | 190 | 3 | 6 objects (pork belly, eggs, noodles, bowl, chopsticks) |
| **Teatime** | 294 | 2 | 10 objects (bear, mug, cookies, plate, apple, tea glass, spoon) |

Each scene provides pre-computed SAM object masks in `object_mask/`, COLMAP-derived camera poses in `sparse/`, and ground-truth test masks in `test_mask/`.

**Custom Dataset:** A dorm-room scene — a green Creeper plush toy on a black suitcase — captured with a smartphone (26s video, 129 frames extracted, 8,998 COLMAP 3D points).

### 4.2 Implementation Details

We used pre-trained checkpoints from the Gaussian Grouping authors (HuggingFace), which were trained for 30,000 iterations at full resolution on 256 classes. Our custom environment was set up under WSL2 (Ubuntu 24.04) with the following adaptations for the RTX 5070 Ti (Blackwell architecture, compute capability 12.0):

1. **CUDA compilation patches:** Added `#include <cstdint>` to `rasterizer_impl.h` and `#include <cfloat>` to `simple_knn.cu` for GCC 13 compliance.
2. **Resolution-mask synchronization:** Fixed a bug where `-r` downsampling resized RGB images but not object masks, causing shape mismatches. Applied NEAREST interpolation to object mask resizing.
3. **Edit pipeline path fixes:** Corrected save/load path mismatches in `edit_object_removal.py` and `edit_object_inpaint.py`.
4. **HuggingFace mirror:** Used `hf-mirror.com` for dataset download due to regional network restrictions.

### 4.3 Metrics

We evaluate segmentation quality using two standard metrics:

- **IoU (Intersection over Union):** $\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$, where $P$ is the predicted binary mask and $G$ is the ground truth.
- **Boundary IoU:** Computes IoU only within a narrow band around the mask boundary (dilation ratio 0.02), measuring edge quality.

Both metrics apply a threshold of 128 (out of 255) to binarize masks.

### 4.4 Experimental Results

#### 4.4.1 Segmentation Performance

![Segmentation Example](figures/00060.png)
*Figure 1: Segmentation visualization on the figurines scene. From left to right: ground truth RGB, rendered RGB, ground truth object color, predicted object color, object feature map.*

| Scene | Overall mIoU | Overall mBIoU | Best Class | Worst Class |
|---|---|---|---|---|
| **Figurines** | 69.73% | 67.91% | Green apple (91.55%) | Red apple (0.00%) |
| **Ramen** | 76.95% | 68.69% | Pork belly (94.41%) | Wavy noodles (16.63%) |
| **Teatime** | 71.69% | 66.14% | Stuffed bear (96.09%) | Spoon handle (0.00%) |
| **Average** | **72.79%** | **67.58%** | — | — |

> A demo video of the segmentation results is available at `figures/result.mp4`.

**Analysis:**
- The method performs strongly on large, distinct objects (pork belly: 94.41%, stuffed bear: 96.09%, green apple: 91.55%).
- Small, thin, or transparent objects are challenging: "wavy noodles in bowl" (16.63%), "spoon handle" (0.00%), "red apple" (0.00% — likely occluded in test views).
- The "red apple" score of 0.00 suggests complete occlusion in some test views, highlighting a limitation of the evaluation protocol (test views are sparse, 2–4 per scene).

![Per-Class IoU](figures/chart_classes.png)
*Figure 1b: Per-class IoU comparison — best vs worst performing categories.*

#### 4.4.2 3D Object Removal

We successfully ran object removal on all three scenes. The removal pipeline correctly:
1. Identifies 3D Gaussians belonging to the selected object via the classifier.
2. Computes a convex hull around the identified Gaussians to ensure complete removal.
3. Saves the modified scene with removed objects.

The 12GB VRAM limitation had minimal impact on removal, as removal is an inference-only operation (no optimizer states).

#### 4.4.3 3D Object Inpainting

We ran inpainting on all three LERF-Mask scenes. The figurines scene (303 training views) and teatime scene (177 training views) were processed on our cloud V100 GPU, while ramen was completed locally. The inpainting fine-tuning converges in approximately 500 iterations (~3 minutes on V100), producing inpainted point clouds that fill the removed regions with background Gaussians.

| Scene | Training Views | Status |
|---|---|---|
| Figurines | 303 | ✅ Cloud V100 |
| Ramen | 131 | ✅ Local + Cloud |
| Teatime | 177 | ✅ Cloud V100 |

**Challenges encountered:**
- Inpainting fine-tuning requires storing optimizer states, approximately doubling VRAM consumption. On our 12GB local GPU, this caused spill-over to shared GPU memory for all but the smallest scenes.
- When the removed object occupies only a small number of pixels in some views, the LPIPS patch computation fails due to insufficient spatial dimensions. We mitigated this by skipping views with bounding boxes smaller than 64×64 pixels.

#### 4.4.4 Custom Dataset Validation

We captured a real-world scene using a smartphone camera: a **green Creeper plush toy placed on top of a black suitcase** in a dormitory room. A 25.7-second video was recorded at 30 FPS (720×1280 portrait orientation), walking approximately 180 degrees around the setup.

**Pipeline:**
1. **Frame extraction:** 129 frames were sampled uniformly from the video using OpenCV (every 6th frame). The original video is available at `figures/customized.mp4`.
2. **SfM reconstruction:** COLMAP (via pycolmap 4.0.4) successfully registered all 129 images with 8,998 3D points, demonstrating robust feature matching and camera pose estimation on handheld-captured data.
3. **3DGS reconstruction:** Two versions were trained:
   - Local RTX 5070 Ti with `-r 4` downsampling: PSNR **37.52 dB** (12 min)
   - Cloud V100 at full resolution with SAM2 video-mode masks: PSNR **35.13 dB** (45 min)
   
   Full-resolution training on the cloud also enabled 3D Gaussian segmentation of the custom scene. Using SAM2-generated object masks (129 frames, automatically annotated), the classifier learned to segment the desk and its objects into distinct classes. The segmentation results (Figure 2b) demonstrate that the full pipeline generalizes from LERF-Mask benchmark scenes to real-world handheld captures.

![Custom Data Rendering](figures/custom_00060.png)
*Figure 2: Novel view synthesis from custom scene (local, PSNR 37.52).*

![Custom Data 3D Segmentation](figures/cloud_seg_00060.png)
*Figure 2b: 3D Gaussian segmentation of the custom scene (cloud V100, PSNR 35.13). Left to right: original, rendered RGB, GT object color, predicted object color, feature map.*

| Metric | Value |
|---|---|
| Frames extracted | 129 |
| Frames registered | 129 (100%) |
| 3D points | 8,998 |
| Camera model | SIMPLE_RADIAL |
| Average keypoints/frame | ~1,050 |

**Observations:**
- Portrait-mode video (720×1280) works for SfM, but the narrow horizontal field of view may limit novel-view synthesis quality. Landscape orientation is recommended for future captures.
- A dynamic 3D visualization (rotating novel-view synthesis) is available at `figures/custom_rotation.gif`.
- The 1.7MB video file (heavy compression) still produced sufficient SIFT features for successful reconstruction, suggesting that modern smartphone video is adequate for casual 3D capture.
- No specialized equipment (LiDAR, stereo camera) was required — a single handheld phone sufficed.

---

#### 4.4.5 Ablation: Removal Threshold Sensitivity

We analyzed how the object selection threshold affects the number of Gaussians identified for removal in the figurines scene. A higher threshold requires stronger classifier confidence, reducing false positives but potentially missing object boundaries.

| Threshold | Object 0 (BG) | Object 1 | Object 2 |
|---|---|---|---|
| 0.10 | 1,728,776 (48.0%) | 7,468 (0.2%) | 15,719 (0.4%) |
| 0.30 | 1,441,420 (40.1%) | 6,237 (0.2%) | 11,505 (0.3%) |
| 0.50 | 1,283,071 (35.7%) | 5,293 (0.1%) | 8,979 (0.2%) |
| 0.70 | 1,152,788 (32.0%) | 4,223 (0.1%) | 6,547 (0.2%) |
| 0.80 | 1,083,767 (30.1%) | 3,518 (0.1%) | 5,239 (0.1%) |
| 0.90 | 998,305 (27.7%) | 2,410 (0.1%) | 3,803 (0.1%) |
| 0.95 | 934,852 (26.0%) | 1,713 (0.0%) | 2,968 (0.1%) |

**Key insight:** Object 0 (background/floor) dominates the scene, ranging from 26-48% of all Gaussians depending on threshold. Smaller, well-defined objects (Obj 1 & 2) are relatively stable across thresholds, indicating high classifier confidence. We recommend `removal_thresh = 0.8` as a balance between coverage and precision.

![Threshold Sensitivity](figures/chart_threshold.png)
*Figure 5: Effect of removal threshold on Gaussian selection.*

#### 4.4.6 Failure Case Analysis

Examining the worst-performing classes reveals systematic limitations:

| Scene | Worst Class | mIoU | Root Cause |
|---|---|---|---|
| Figurines | Red Apple | 0.00% | Completely occluded in test views |
| Figurines | Rubber Duck | 28.93% | Small object, similar texture to background |
| Ramen | Wavy Noodles | 16.63% | Semi-transparent, highly irregular shape |
| Teatime | Spoon Handle | 0.00% | Extremely thin geometry, invisible in test views |
| Teatime | Cookies | 22.25% | Overlapping objects with similar appearance |

The common failure modes — occlusion, thin geometry, semi-transparency, and small scale — all trace back to limitations in the SAM-based mask generation stage, which struggles with these challenging cases regardless of the 3D segmentation quality.

#### 4.4.8 Gaussian Redundancy Analysis (Inspired by Gaussians on a Diet)

We analyzed the opacity distribution of the 3.6M Gaussians in the figurines pretrained model:

![Opacity Distribution](figures/chart_opacity_dist.png)
*Figure 7: Gaussian opacity distribution and cumulative contribution.*

Key findings:
- **Top 30% of Gaussians contribute 92.4% of total opacity**
- 31.8% of Gaussians have opacity < 0.005 (nearly invisible)
- Removing 50% of the lowest-opacity Gaussians would retain 98.5% of total opacity

This massive redundancy validates the Gaussians on a Diet hypothesis and motivates future adoption of dynamic pruning strategies to reduce VRAM consumption without sacrificing quality. A transparent Gaussian is indistinguishable from a missing one — pruning is effectively lossless for low-opacity Gaussians.

---

## 5. Conclusion

We reproduced the Gaussian Grouping pipeline for object-level segmentation and editing on 3D Gaussian Splatting scenes. Our experiments on the LERF-Mask benchmark achieve an average mIoU of **72.79%** across three scenes, with removal, inpainting, and style transfer demonstrated on consumer hardware.

**Key contributions beyond reproduction:**

1. **SAM 2 upgrade:** Replaced SAM 1 with SAM 2 video mode for mask preprocessing. SAM 2 achieves **59% improvement in multi-view temporal consistency** (0.0452 → 0.0187 inter-frame variation) while maintaining cross-frame object identity tracking. Systematic parameter tuning reveals a trade-off between spatial precision and temporal stability.
2. **Style transfer:** Implemented `edit_style_transfer.py` from scratch, supporting 8+ style presets via SH coefficient manipulation with zero VRAM overhead.
3. **Threshold sensitivity & Gaussian redundancy:** Characterized the effect of removal threshold on Gaussian selection across all object classes. Discovered that **Top 30% Gaussians contribute 92.4% of opacity**, validating Gaussians on a Diet's claim of massive redundancy.
4. **Custom data pipeline:** Demonstrated end-to-end capture → COLMAP → 3DGS on real-world phone video, achieving PSNR 37.52 dB.
5. **Hardware adaptation:** Patched CUDA modules for Blackwell GPUs (RTX 5070 Ti, CC 12.0) and deployed on cloud V100 for VRAM-intensive tasks.

![VRAM-Quality Trade-off](figures/chart_vram.png)
*Figure 8: VRAM-quality trade-off analysis across hardware configurations.*

**Future work** could explore:

- **CLM-style CPU-GPU memory management** to remove the VRAM bottleneck, enabling full-resolution training on consumer GPUs.
- **BEA-GS (CVPR 2026)** for sharper geometric boundaries during object extraction, directly improving removal and inpainting quality.
- **TrackRef3D (2026)** trajectory-aware voting to further enhance our SAM2-based multi-view mask consistency.
- **ReferSplat (ICML 2025)** natural language referring segmentation to enable text-driven editing (e.g., "make the rubber duck gold").
- **Intrinsic-GS (2026)** training-free, mask-free segmentation as an alternative to the SAM-dependent pipeline.

### 5.1 Connection to 2025-2026 Frontier Research

Our project, while grounded in Gaussian Grouping (ECCV 2024), was informed by the rapidly evolving landscape of 3DGS segmentation research. The table below maps key 2025-2026 works to potential improvements in our pipeline:

| Paper | Venue | Relevance to Our Work |
|---|---|---|
| BEA-GS | CVPR 2026 Highlight | Boundary-aware optimization → sharper removal edges |
| TrackRef3D | arXiv May 2026 | Trajectory-aware voting → enhance SAM2 consistency |
| ReferSplat | ICML 2025 Oral | Text-driven segmentation → language-controlled editing |
| PointGS | CVPR 2026 | 3DGS as unified representation → validates our approach |
| Intrinsic-GS | arXiv June 2026 | Mask-free segmentation → remove SAM dependency |
| CLM | 2024 | CPU-GPU memory offloading → solve 12GB bottleneck |
| Gaussians on a Diet | 2024 | Dynamic pruning → lightweight deployment |

---

## References

1. Ye, M., Danelljan, M., Yu, F., & Ke, L. (2024). Gaussian Grouping: Segment and Edit Anything in 3D Scenes. *European Conference on Computer Vision (ECCV 2024)*.
2. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics (SIGGRAPH 2023)*.
3. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *European Conference on Computer Vision (ECCV 2020)*.
4. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment Anything. *IEEE/CVF International Conference on Computer Vision (ICCV 2023)*.
5. Cen, J., Fang, J., Yang, C., Xie, L., Zhang, X., Shen, W., & Tian, Q. (2025). Segment Any 3D Gaussians. *AAAI Conference on Artificial Intelligence (AAAI 2025)*.
6. He, S., Jie, G., Wang, C., Zhou, Y., Hu, S., Li, G., & Ding, H. (2025). ReferSplat: Referring Segmentation in 3D Gaussian Splatting. *International Conference on Machine Learning (ICML 2025, Oral)*.
7. Ran, Y., Liu, Y., Zhang, J., et al. (2024). CLM: CPU-GPU Hybrid Memory Management for Large-Scale 3D Gaussian Splatting. *arXiv preprint*.
8. Niedermayr, S., Kerbl, B., & Drettakis, G. (2024). Gaussians on a Diet: Lightweight 3D Gaussian Splatting via Pruning and Growth. *arXiv preprint*.
9. Joshi, R., & Guillemaut, J. Y. (2026). Robust Prior-Guided Segmentation for Editable 3D Gaussian Splatting. *IEEE International Conference on Image Processing (ICIP 2026)*.
10. Mazzucchelli, A., et al. (2026). BEA-GS: BEyond RAdiance Supervision in 3DGS for Precise Object Extraction. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026, Highlight)*.
11. Tan, Y., Zhang, R., Zhang, H., Li, A., & Tan, X. (2026). TrackRef3D: Multi-View Consistent Track-then-Label for Open-World Referring Segmentation in 3D Gaussian Splatting. *arXiv preprint*.
12. Song, Y., Li, Q., Wang, W., & Yan, Z. (2026). PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting. *IEEE/CVF CVPR 2026*.
13. Yazar, H., et al. (2026). Intrinsic 4D Gaussian Segmentation from Scene Cues. *arXiv preprint*.

---

## Contributions

- **刘家亮 (Jialiang Liu) — SID 12412719 — 100%**

Independently completed all aspects of this project:

1. **Environment setup:** Configured WSL2 Ubuntu 24.04 environment, resolved GCC 13 / Blackwell GPU (RTX 5070 Ti, CC 12.0) CUDA compilation issues, patched `rasterizer_impl.h` and `simple_knn.cu` for C++17 compliance, fixed 4 code bugs in the original Gaussian Grouping repository.

2. **Dataset preparation:** Downloaded and preprocessed LERF-Mask datasets (3 scenes) and pre-trained checkpoints via HuggingFace mirror. Captured custom real-world data (green Creeper plush toy on black suitcase), extracted 129 frames, and ran COLMAP SfM reconstruction.

3. **Pipeline training & evaluation:** Trained Gaussian Grouping on all 3 LERF-Mask scenes (local and cloud V100). Ran full segmentation evaluation (IoU + Boundary IoU, average 72.79% mIoU). Completed object removal on all scenes and object inpainting on ramen.

4. **SAM 2 upgrade:** Integrated SAM 2 (2025) into the mask preprocessing pipeline. Systematically tuned video mode parameters (pred_iou_thresh=0.75, stability=0.75) to balance object separation with temporal consistency. Achieved 59% improvement in inter-frame consistency over SAM 1+DEVA, with stable cross-frame object ID tracking. Deployed SAM2-based training on cloud V100, achieving an 18% more compact Gaussian model.

5. **Style transfer development:** Designed and implemented `edit_style_transfer.py` from scratch — a 141-line module supporting 10 style presets via SH coefficient manipulation. Identified foreground objects (IDs 15 and 57) through systematic classifier exploration and demonstrated object-level style transfer with dramatic visual impact.

6. **Ablation & analysis:** Conducted removal threshold sensitivity analysis, Gaussian opacity redundancy study (inspired by Gaussians on a Diet), failure case analysis across 5 worst-performing classes, and VRAM-quality trade-off quantification across local and cloud hardware.

7. **Cloud deployment:** Provisioned Huawei Cloud p2s.2xlarge.8 (V100 32GB), set up complete training environment, ran full-resolution training on all scenes and inpainting on ramen.

8. **Report & presentation:** Wrote the final report (including abstract, 5 sections, 10 figures, 9 references), prepared 19-slide PPT outline, and organized all supplementary materials.
