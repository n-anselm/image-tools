#!/usr/bin/env python3
"""
Created by OpenCode
Initialized: 2026-05-19
Updated: 2026-05-19

Image Stitcher CLI — Assemble overlapping images into a composite image.

Uses OpenCV's built-in Stitcher (feature-based with homography estimation
and multi-band blending) to automatically detect overlaps and composite images.

Supports two modes:
  PANORAMA  — allows rotation, best for camera panoramas
  SCANS     — assumes pure translation, best for flat screenshots

Usage:
    python stitch_images.py input_folder/ [options]

Output is always saved to input_folder/stitched/stitched.png.

Examples:
    python stitch_images.py ./screenshots/
    python stitch_images.py ./screenshots/ --mode scans
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Light-weight setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stitch_images")

SUPPORTED_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"
}

REQUIRED_PACKAGES: dict[str, str] = {
    "cv2": "opencv-python",
    "numpy": "numpy",
}

# ---------------------------------------------------------------------------
# 2. CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stitch_images.py",
        description="Assemble overlapping images using OpenCV Stitcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  panorama (default)
    OpenCV's default stitcher. Allows slight rotation and scale changes.
    Best when images might have perspective differences.

  scans
    Optimized for flat documents/screenshots with pure translation.
    Faster and often more accurate for screenshot grids.

Output is always saved to input_folder/stitched/stitched.png.

Examples:
  %(prog)s ./screenshots/
  %(prog)s ./screenshots/ --mode scans
        """,
    )
    parser.add_argument("input", help="Folder containing images to stitch")
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Interactive configuration mode",
    )
    parser.add_argument(
        "--mode", choices=["panorama", "scans"], default="panorama",
        help="Stitcher mode (default: panorama)",
    )
    return parser


# ---------------------------------------------------------------------------
# 3. Venv & deps
# ---------------------------------------------------------------------------


def _ask_user(question: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            resp = input(f"{question} [{hint}]: ").strip().lower()
        except EOFError:
            return default
        if not resp:
            return default
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("  Please answer 'y' or 'n'.")


def _in_virtual_env() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or getattr(sys, "base_prefix", None) != sys.prefix
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def _get_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv() -> None:
    if _in_virtual_env():
        return
    venv_path = Path(__file__).resolve().parent / ".venv"
    if venv_path.exists() and venv_path.is_dir():
        venv_python = _get_venv_python(venv_path)
        if venv_python.exists():
            script_path = Path(__file__).resolve()
            os.execv(str(venv_python), [str(venv_python), str(script_path)] + sys.argv[1:])
    logger.warning("Not in a virtual environment.")
    if not _ask_user("Create a virtual environment?", default=True):
        sys.exit(0)
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    venv_python = _get_venv_python(venv_path)
    script_path = Path(__file__).resolve()
    os.execv(str(venv_python), [str(venv_python), str(script_path)] + sys.argv[1:])


def _ensure_dependencies() -> None:
    missing = []
    for mod, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if not missing:
        return
    logger.warning("Missing packages: %s", ", ".join(missing))
    if not _ask_user("Install missing dependencies now?", default=True):
        sys.exit(0)
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)


def ensure_environment() -> None:
    _ensure_venv()
    _ensure_dependencies()


if "--help" in sys.argv or "-h" in sys.argv:
    _build_parser().print_help()
    sys.exit(0)

ensure_environment()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# 4. Image loading
# ---------------------------------------------------------------------------


def load_images(folder: Path) -> list[np.ndarray]:
    """Load all supported images from folder. Returns sorted BGR array list."""
    files: set[Path] = set()
    for ext in SUPPORTED_EXTENSIONS:
        files.update(folder.glob(f"*{ext}"))
        files.update(folder.glob(f"*{ext.upper()}"))
    result: list[np.ndarray] = []
    for p in sorted(files):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            result.append(img)
            logger.debug("Loaded %s (%dx%d)", p.name, img.shape[1], img.shape[0])
        else:
            logger.warning("Failed to load %s", p)
    logger.info("Loaded %d images from %s", len(result), folder)
    return result


# ---------------------------------------------------------------------------
# 5. Stitching
# ---------------------------------------------------------------------------


def stitch_images(images: list[np.ndarray], mode: str) -> np.ndarray | None:
    """
    Stitch images using OpenCV Stitcher.

    Args:
        images: List of BGR images.
        mode: 'panorama' or 'scans'.

    Returns:
        Stitched BGR image or None on failure.
    """
    if mode == "scans":
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    else:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

    logger.info("Running OpenCV Stitcher in %s mode...", mode.upper())
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return pano

    error_codes = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images (try >= 2)",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Could not estimate alignment — images may not overlap",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed",
    }
    msg = error_codes.get(status, f"Unknown stitcher error ({status})")
    logger.error("Stitching failed: %s", msg)
    return None


# ---------------------------------------------------------------------------
# 6. Interactive mode
# ---------------------------------------------------------------------------


def interactive_configure(args: argparse.Namespace) -> argparse.Namespace:
    print("\n" + "=" * 60)
    print("  Image Stitcher — Interactive Configuration")
    print("=" * 60 + "\n")

    print("Which stitcher mode do you want to use?")
    print("  1. panorama — default, allows slight rotation/scale")
    print("  2. scans    — optimized for flat screenshots (pure translation)")
    while True:
        c = input("Enter choice [1-2, default: 1]: ").strip()
        if not c or c == "1":
            args.mode = "panorama"
            break
        if c == "2":
            args.mode = "scans"
            break
        print("  Invalid. Enter 1 or 2.")

    print("\n" + "=" * 60)
    print("  Configuration complete!")
    print("=" * 60 + "\n")
    return args


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.interactive:
        args = interactive_configure(args)

    input_path = Path(args.input).resolve()
    if not input_path.exists() or not input_path.is_dir():
        logger.error("Input must be an existing directory: %s", input_path)
        return 1

    images = load_images(input_path)
    if len(images) < 2:
        logger.error("Need at least 2 images to stitch. Found %d.", len(images))
        return 1

    # Output always goes to input/stitched/stitched.png
    output_path = input_path / "stitched" / "stitched.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Output: %s", output_path)
    logger.info("Mode: %s", args.mode)

    result = stitch_images(images, args.mode)
    if result is None:
        logger.info(
            "Tips:\n"
            "  • Ensure images share some overlapping content.\n"
            "  • Try the other --mode if one fails.\n"
            "  • OpenCV Stitcher requires distinct features; blank areas may cause issues."
        )
        return 1

    cv2.imwrite(str(output_path), result)
    logger.info("Saved stitched image: %s (%dx%d)", output_path, result.shape[1], result.shape[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
