# Geospatial AI Segmentation Pipeline

A multi-model geospatial inference pipeline that detects and segments **buildings, roads, and water bodies** in high-resolution GeoTIFF imagery. It combines **YOLO** and **RT-DETR** object detection with **SAM 2** segmentation refinement, and exports results as YOLO-format annotations for downstream model training.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Multi-model inference**: YOLO and RT-DETR detection, refined by SAM 2 segmentation
- **Large-raster support**: 1024 × 1024 overlapping image tiling for high-resolution GeoTIFFs
- **Post-processing**: confidence-based filtering and mask merging across tile boundaries
- **Optimized performance**: GPU execution, tiled processing, and memory-aware array handling
- **Annotation export**: automated segmentation-mask conversion and YOLO-format annotation generation

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | 85% |
| Precision | 90% |
| Recall | 80% |

Evaluated on a labeled evaluation dataset.

| Optimization | Improvement vs. baseline |
|--------------|--------------------------|
| Processing time | 70% reduction |
| Peak memory usage | 60% reduction |
| Manual annotation effort | 70% reduction |

The pipeline generated **5,000+ annotations** for downstream model training.

## Pipeline Overview

```
GeoTIFF Input
     │
     ▼
1024 × 1024 Overlapping Tiling
     │
     ▼
YOLO / RT-DETR Detection
     │
     ▼
Confidence-Based Filtering
     │
     ▼
SAM 2 Segmentation Refinement
     │
     ▼
Mask Merging Across Tiles
     │
     ▼
YOLO-Format Annotation Export
```

## Tech Stack

Python, PyTorch, YOLO, RT-DETR, SAM 2, OpenCV, NumPy

## Installation

```bash
git clone https://github.com/trobcooper/geotiff-feature-extraction.git
cd geotiff-feature-extraction

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> A CUDA-capable GPU is recommended for large-raster inference.

## Usage

```bash
python <main_script>.py --input <path/to/image.tif> --output <path/to/output_dir>
```

| Argument | Description |
|----------|-------------|
| `--input` | Path to the input GeoTIFF image |
| `--output` | Directory for masks and annotations |
| `--conf` | Confidence threshold for filtering detections |

## Project Structure

```
geotiff-feature-extraction/
├── <main_script>.py        # Entry point for the inference pipeline
├── requirements.txt        # Python dependencies
├── models/                 # Model weights (YOLO, RT-DETR, SAM 2)
├── outputs/                # Generated masks and YOLO annotations
└── README.md
```

## Output

- Segmentation masks for buildings, roads, and water bodies
- YOLO-format annotation files ready for model training

## Author

**Chaitanya Chintalapati**
[LinkedIn](https://www.linkedin.com/in/chaitanya-chintalapati-761820309/) | [GitHub](https://github.com/trobcooper)
