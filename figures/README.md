# Supplementary Materials

## Animations
| File | Description |
|---|---|
| `style_transfer_rotation.mp4` | 4-style rotating viewpoint comparison (Original/Red/Gold/Blue) |
| `custom_rotation.mp4` | Custom scene 360 degree novel-view synthesis |
| `custom_seg_cloud.mp4` | Custom data 3D Gaussian segmentation (cloud V100) |

## Charts
| File | Description |
|---|---|
| `chart_threshold.png` | Removal threshold sensitivity analysis |
| `chart_vram.png` | VRAM-quality trade-off (local vs cloud) |
| `chart_classes.png` | Per-class IoU: best vs worst |
| `chart_opacity_dist.png` | Gaussian opacity distribution + cumulative contribution |
| `chart_sam2_consistency.png` | SAM2 video mode: 59 percent temporal consistency improvement |
| `chart_frame_variation.png` | Frame-by-frame mask variation (SAM1 vs SAM2) |

## Segmentation
| File | Description |
|---|---|
| `00060.png` | LERF-Mask figurines segmentation visualization (5-column) |
| `cloud_seg_00000.png` | Custom data 3D Gaussian segmentation (view 0) |
| `cloud_seg_00030.png` | Custom data 3D Gaussian segmentation (view 30) |
| `cloud_seg_00060.png` | Custom data 3D Gaussian segmentation (view 60) |
| `cloud_seg_00090.png` | Custom data 3D Gaussian segmentation (view 90) |
| `custom_multiview_seg.png` | Multi-view SAM2 segmentation on custom data |
| `custom_sam2_seg.png` | SAM2 mask overlay on custom data |
| `sam_compare_video.png` | SAM1 vs SAM2 video mode comparison (3 frames) |

## Style Transfer
| File | Description |
|---|---|
| `id15_original_view80.png` | Original scene (view 80) |
| `id15_red_view80.png` | Object ID 15 colorized to red |
| `id57_original_front.png` | Original scene (front view) |
| `id57_blue_front.png` | Object ID 57 colorized to blue |
| `sequential_editing.png` | Multi-object independent editing: Original | ID15 Red | ID57 Blue |

## Object Removal
| File | Description |
|---|---|
| `removal_before.png` | Original figurines scene |
| `removal_fixed.png` | ID 32 removed via convex hull (12 percent Gaussians deleted) |
| `removal_id57_convex.png` | ID 57 removed via convex hull (precision removal) |

## Dataset
| File | Description |
|---|---|
| `dataset_figurines.jpg` | LERF-Mask figurines scene |
| `dataset_ramen.jpg` | LERF-Mask ramen scene |
| `dataset_teatime.jpg` | LERF-Mask teatime scene |

## Custom Data
| File | Description |
|---|---|
| `custom_rotation.mp4` | Custom scene novel-view synthesis rotation |
| `custom_seg_cloud.mp4` | Custom data 3D Gaussian segmentation video |
| `custom_multiview_seg.png` | Multi-view SAM2 segmentation on custom data |
| `custom_sam2_seg.png` | SAM2 mask overlay on custom data |
