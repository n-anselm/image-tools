# image-tools

A personal experimental toolkit for exploring what can be extracted and created from images — depth maps, edge detection, panoramas, anaglyph 3D, and more. Each script is a self-contained CLI tool with consistent behavior: auto venv detection, dependency management, interactive mode, and batch processing support.

## Scripts

| Script | Description |
|--------|-------------|
| `depth_map.py` | Generate depth maps from images using MiDaS DPT models. Supports grayscale or colormap output, GPU/CPU auto-selection, batch processing, and multiple colormaps. |
| `edge_detection.py` | Detect edges in images using OpenCV's Canny algorithm. User-friendly threshold controls (0–100), grayscale output, and batch folder processing. |
| `invert_color.py` | Invert the colors of an image or all images in a folder. Fast, simple, and supports all common image formats. |
| `stitch_images.py` | Assemble overlapping images into a composite panorama. Uses OpenCV's built-in Stitcher with two modes: `panorama` (allows rotation) and `scans` (pure translation). |
| `anaglyph_3d.py` | Create red-cyan anaglyph 3D images from a photo and its depth map. Auto-detects `<stem>-depthmap.png` or prompts to generate one on demand. Adjustable parallax strength. |
| `exif_extractor.py` | Extract camera metadata (focal length, sensor size, GPS, exposure, etc.) from images into structured JSON. Computes camera intrinsics when possible. Optional Markdown export in interactive mode. Batch folder support. |
| `point_cloud.py` | Generate a colored 3D point cloud (.ply) from a photo and its depth map. Auto-detects depth map and camera intrinsics, with configurable resolution. Uses pinhole camera back-projection. |

## Quick Start

All scripts are standalone and handle their own environment:

```bash
python depth_map.py photo.png
python edge_detection.py ./photos/
python anaglyph_3d.py photo.png -i
python point_cloud.py photo.png --resolution half
```

Run any script with `--help` for full usage details.

![image](https://github.com/n-anselm/image-tools/blob/main/assets/image-tools-info.jpg)
