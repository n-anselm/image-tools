#!/usr/bin/env python3
"""
Created by OpenCode
Initialized: 2026-05-18
Updated: 2026-05-18

Invert colors of an image or all images in a folder.

Supports PNG, JPG/JPEG, BMP, TIFF, and WebP formats.

Usage:
    python invert_color.py input.png [output.png]
    python invert_color.py /path/to/images [/path/to/output]

Output rules:
  • Single file input + no output  -> <name>_inverted.<ext> in same folder
  • Single file input + output     -> written to specified path
  • Folder input    + no output    -> creates <input>/inverted/ folder
  • Folder input    + output       -> writes to specified output folder
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("invert_color")

# Supported image extensions (aligned with depth_map.py)
SUPPORTED_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


def invert_image(input_path: Path, output_path: Path) -> bool:
    """
    Invert the colors of a single image and save it.

    Args:
        input_path: Path to the source image.
        output_path: Destination path for the inverted image.

    Returns:
        True on success, False on failure.
    """
    try:
        img = Image.open(input_path)
        img = ImageOps.exif_transpose(img) or img
        img = img.convert("RGB")
        inverted = Image.eval(img, lambda x: 255 - x)

        # Ensure the parent directory exists before saving
        output_path.parent.mkdir(parents=True, exist_ok=True)
        inverted.save(output_path)
        logger.info("Inverted: %s -> %s", input_path, output_path)
        return True
    except Exception as exc:
        logger.error("Failed to invert %s: %s", input_path, exc)
        return False


def collect_images(input_path: Path) -> list[Path]:
    """
    Gather all supported image files from the given path.

    Args:
        input_path: A file or directory path.

    Returns:
        Sorted list of image Paths. Empty if none found.
    """
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path.resolve()]
        logger.error("Unsupported file format: %s", input_path.suffix)
        return []

    if not input_path.is_dir():
        logger.error("Input path does not exist: %s", input_path)
        return []

    images: set[Path] = set()
    for ext in SUPPORTED_EXTENSIONS:
        images.update(input_path.glob(f"*{ext}"))
        images.update(input_path.glob(f"*{ext.upper()}"))

    if not images:
        logger.warning("No images found in: %s", input_path)

    return sorted(p.resolve() for p in images)


def resolve_output_paths(
    input_paths: list[Path], output_arg: str | None, fmt: str
) -> list[Path]:
    """
    Generate output paths for each input image.

    Args:
        input_paths: List of input image paths.
        output_arg: Optional user-provided output path string.
        fmt: Default file extension (without dot).

    Returns:
        List of output paths aligned with input_paths.
    """
    if not input_paths:
        return []

    fmt = fmt.lower().lstrip(".")

    # Single input file -----------------------------------------------------
    if len(input_paths) == 1:
        inp = input_paths[0]
        if output_arg:
            out_raw = Path(output_arg)
            # If user explicitly ended with a path separator, or the path
            # already exists as a directory, treat it as a folder.
            if output_arg.endswith(("/", "\\")) or (
                out_raw.exists() and out_raw.is_dir()
            ):
                out = out_raw / f"{inp.stem}_inverted.{fmt}"
            else:
                out = out_raw.with_suffix(f".{fmt}")
        else:
            out = inp.parent / f"{inp.stem}_inverted.{fmt}"
        return [out]

    # Multiple inputs (folder mode) -----------------------------------------
    if output_arg:
        out_dir = Path(output_arg)
        # If the user passed what looks like a file path for multiple images,
        # that's an error.
        if out_dir.suffix and not (out_dir.exists() and out_dir.is_dir()):
            logger.error(
                "Cannot specify a single file output for multiple input images."
            )
            sys.exit(1)
    else:
        out_dir = input_paths[0].parent / "inverted"

    return [out_dir / f"{p.stem}_inverted.{fmt}" for p in input_paths]


def main() -> int:
    """Parse arguments, collect images, invert and save them."""
    parser = argparse.ArgumentParser(
        description="Invert the colors of an image or all images in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.png
  %(prog)s photo.png my_inverted.png
  %(prog)s ./photos/
  %(prog)s ./photos/ ./inverted_photos/
        """,
    )
    parser.add_argument("input", help="Input image file or folder containing images")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output image file or folder (optional)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        default="png",
        help="Output image format for auto-generated paths (default: png)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    images = collect_images(input_path)
    if not images:
        logger.error("No valid images found to process.")
        return 1

    output_paths = resolve_output_paths(images, args.output, args.format)

    # Summary
    logger.info("Input:  %s", input_path)
    logger.info("Images: %d", len(images))

    success = 0
    for img_in, img_out in zip(images, output_paths):
        if invert_image(img_in, img_out):
            success += 1

    logger.info(
        "Finished. Successfully inverted %d / %d images.",
        success,
        len(images),
    )
    return 0 if success == len(images) else 1


if __name__ == "__main__":
    sys.exit(main())
