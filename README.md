# SUSTech CS308 - Computer Vision 2026 Spring Final Project

## Modernizing 3D Gaussian Segmentation: SAM 2 Integration and VRAM-Efficient Editing Pipeline

**Score: 109/100**

This project reproduces and extends [**Gaussian Grouping**](https://github.com/lkeab/gaussian-grouping) (Ye et al., ECCV 2024) for object-level segmentation and editing on 3D Gaussian Splatting scenes.

### Key Contributions

- **SAM 2 Video Mode Upgrade**: Replaced SAM 1 with SAM 2 video mode for mask preprocessing, achieving **59% improvement** in multi-view temporal consistency with stable cross-frame object tracking.
- **Style Transfer Module**: Implemented custom style transfer via Spherical Harmonics (SH) coefficient manipulation.
- **Comprehensive Ablation Studies**: Threshold sensitivity, Gaussian redundancy analysis, and failure case examination.
- **Custom Real-World Validation**: Tested pipeline on self-captured data achieving PSNR 37.52 dB.
- **CUDA Fixes for Blackwell GPU (RTX 5070 Ti)**: Patched `diff-gaussian-rasterization` and `simple-knn` for GCC 13 strict C++17 compliance (compute capability 12.0).
- **VRAM-Efficient Pipeline**: Deployed on cloud V100 (32GB) for full-resolution experiments.

### Achieved Metrics

| Scene | mIoU |
|-------|------|
| figurines | 80.55% |
| ramen | 66.64% |
| teatime | 71.18% |
| **Average** | **72.79%** |

### Project Structure

```
├── gaussian-grouping/       # Modified codebase (forked from lkeab/gaussian-grouping)
│   ├── train.py             # Training with Identity Encoding
│   ├── render.py            # Segmentation rendering
│   ├── render_lerf_mask.py  # LERF-Mask evaluation rendering
│   ├── edit_object_removal.py
│   ├── edit_object_inpaint.py
│   ├── scene/               # Scene data handling
│   ├── gaussian_renderer/   # Differentiable rasterizer
│   ├── submodules/          # diff-gaussian-rasterization, simple-knn
│   ├── utils/               # Camera utils (patched for mask resize)
│   └── config/              # Training & editing configs
├── figures/                 # Result visualizations
├── report/
│   ├── 26_final.pdf         # Final project report (PDF)
│   └── Report.md            # Report source (Markdown)
└── README.md
```

### Acknowledgments

This project is based on [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) (ECCV 2024) by Ye et al., licensed under Apache 2.0. We thank the authors for their open-source contributions.
