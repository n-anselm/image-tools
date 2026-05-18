#!/usr/bin/env python3
"""
Created by OpenCode
Initialized: 2026-05-18
Updated: 2026-05-18

Edge Detection CLI using OpenCV Canny.

Detect edges in images or all images in a folder using the Canny algorithm.
Thresholds are user-friendly (0-100) and mapped internally to OpenCV's 0-255 range.
Output is grayscale by default.

Usage:
    python edge_detection.py input.png [output.png] [options]
    python edge_detection.py input_folder/ [output_folder/] [options]

Interactive Mode:
    python edge_detection.py input.png -i

Environment:
    The script checks for a virtual environment on startup. If none is detected,
    it asks to create one. If a .venv already exists in the script's directory,
    the script restarts into it automatically. Missing dependencies (opencv-python,
    Pillow, numpy) are checked and can be auto-installed with confirmation.

Examples:
    # Basic usage: generates input-edges.png
    python edge_detection.py photo.png

    # Specify output file
    python edge_detection.py photo.png my_edges.png

    # Process a folder (creates ./photos/edges/ automatically)
    python edge_detection.py ./photos/

    # Adjust sensitivity
    python edge_detection.py photo.png --low 30 --high 70

    # Interactive mode to choose settings
    python edge_detection.py photo.png -i
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Light-weight setup (no heavy imports yet)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edge_detection")

# Supported image extensions (aligned with depth_map.py and invert_color.py)
SUPPORTED_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}

# Map import names to pip package names
REQUIRED_PACKAGES: dict[str, str] = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "Pillow",
}

# Mapping factor: user threshold 0-100 -> OpenCV 0-255
_THRESHOLD_SCALE: float = 255.0 / 100.0


# ---------------------------------------------------------------------------
# 2. CLI argument parsing (--help works before any heavy setup)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser with full help text."""
    parser = argparse.ArgumentParser(
        prog="edge_detection.py",
        description="Detect edges in images using OpenCV Canny.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Threshold values are on a user-friendly 0-100 scale and are mapped to
OpenCV's 0-255 range internally (multiply by 2.55).

Interactive mode (-i) lets you configure settings step-by-step in the terminal.

Output rules:
  • Single file input + no output  ->  <name>-edges.<ext> in same folder
  • Single file input + output     ->  written to specified path
  • Folder input    + no output    ->  creates <input>/edges/ folder
  • Folder input    + output       ->  writes to specified output folder

Examples:
  %(prog)s photo.png
  %(prog)s photo.png edges.png --low 30 --high 70
  %(prog)s ./photos/ ./edges/ --format png
  %(prog)s photo.png -i
        """,
    )

    parser.add_argument(
        "input",
        help="Input image file or folder containing images",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output image file or folder (optional)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enter interactive mode to configure settings step-by-step",
    )

    # Edge detection settings
    parser.add_argument(
        "--low",
        type=int,
        default=20,
        metavar="0-100",
        help="Canny low threshold, 0-100 scale (default: 20, maps to ~51 in OpenCV)",
    )
    parser.add_argument(
        "--high",
        type=int,
        default=60,
        metavar="0-100",
        help="Canny high threshold, 0-100 scale (default: 60, maps to ~153 in OpenCV)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        default="png",
        help="Output image format for auto-generated paths (default: png)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        default=True,
        help="Output grayscale edges (default: true)",
    )

    return parser


# ---------------------------------------------------------------------------
# 3. Virtual Environment & Dependency Management
# ---------------------------------------------------------------------------


def _ask_user(question: str, default: bool = True) -> bool:
    """Prompt the user for a yes/no answer in the terminal."""
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            response = input(f"{question} [{hint}]: ").strip().lower()
        except EOFError:
            logger.info(
                "Non-interactive environment detected. Using default: %s", default
            )
            return default
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("  Please answer 'y' or 'n'.")


def _in_virtual_env() -> bool:
    """Return True if the current interpreter is running inside a venv."""
    return (
        hasattr(sys, "real_prefix")
        or (getattr(sys, "base_prefix", None) != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def _get_venv_python(venv_dir: Path) -> Path:
    """Return the python executable inside the given venv."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv() -> None:
    """
    Ensure the script runs inside a virtual environment.

    If the current interpreter is already inside a venv, do nothing.
    If a .venv directory already exists next to the script, restart into it
    automatically (no prompt needed).
    Otherwise, ask the user to create a new venv and restart into it.
    """
    if _in_virtual_env():
        logger.debug("Already running inside a virtual environment.")
        return

    venv_path = Path(__file__).resolve().parent / ".venv"

    # A .venv exists but the user didn't activate it — jump in automatically.
    if venv_path.exists() and venv_path.is_dir():
        logger.info("Found existing virtual environment: %s", venv_path)
        venv_python = _get_venv_python(venv_path)
        if venv_python.exists():
            logger.info("Restarting script inside the virtual environment...")
            script_path = Path(__file__).resolve()
            os.execv(
                str(venv_python),
                [str(venv_python), str(script_path)] + sys.argv[1:],
            )
        else:
            logger.warning(
                "Virtual environment exists but python executable not found: %s",
                venv_python,
            )
            # Fall through to ask if we should recreate it.

    logger.warning("You are NOT running inside a virtual environment.")
    logger.info("It is strongly recommended to use a venv to avoid dependency conflicts.")

    if not _ask_user("Create a virtual environment?", default=True):
        logger.info("Exiting. Please activate a virtual environment and run again.")
        sys.exit(0)

    logger.info("Creating virtual environment at: %s", venv_path)

    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to create virtual environment: %s", exc)
        sys.exit(1)

    venv_python = _get_venv_python(venv_path)
    if not venv_python.exists():
        logger.error(
            "Virtual environment created but python executable not found: %s",
            venv_python,
        )
        sys.exit(1)

    logger.info("Restarting script inside the virtual environment...")
    script_path = Path(__file__).resolve()
    os.execv(str(venv_python), [str(venv_python), str(script_path)] + sys.argv[1:])


def _check_dependencies() -> list[str]:
    """Check for missing packages and return their pip install names."""
    missing: list[str] = []
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    return missing


def _install_dependencies(packages: list[str]) -> None:
    """Install missing packages using pip in the current environment."""
    logger.info("Installing: %s", ", ".join(packages))
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    try:
        subprocess.check_call(cmd)
        logger.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to install dependencies: %s", exc)
        sys.exit(1)


def _ensure_dependencies() -> None:
    """
    Ensure required packages are installed.
    Asks for user confirmation before downloading anything.
    """
    missing = _check_dependencies()
    if not missing:
        return

    logger.warning("Missing required packages: %s", ", ".join(missing))
    logger.info("These packages are needed to run edge detection.")
    if not _ask_user("Install missing dependencies now?", default=True):
        logger.info(
            "Exiting. Install manually with: pip install %s",
            " ".join(missing),
        )
        sys.exit(0)
    _install_dependencies(missing)


def ensure_environment() -> None:
    """Top-level environment gate: venv first, then dependencies."""
    _ensure_venv()
    _ensure_dependencies()


# ---------------------------------------------------------------------------
# 4. Heavy imports (only after environment is confirmed)
# ---------------------------------------------------------------------------

# If the user only wants help, print it immediately without any heavy setup.
if "--help" in sys.argv or "-h" in sys.argv:
    _build_parser().print_help()
    sys.exit(0)

ensure_environment()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402


# ---------------------------------------------------------------------------
# 5. Core edge-detection logic (refactored from original GUI plugin)
# ---------------------------------------------------------------------------


def detect_edges(
    image_path: Path,
    output_path: Path,
    low_threshold: int,
    high_threshold: int,
    grayscale: bool = True,
) -> bool:
    """
    Detect edges in a single image using OpenCV Canny and save the result.

    Args:
        image_path: Path to the source image.
        output_path: Destination path for the edge-detected image.
        low_threshold: Canny low threshold (already mapped to OpenCV 0-255).
        high_threshold: Canny high threshold (already mapped to OpenCV 0-255).
        grayscale: If True, output a single-channel grayscale image.

    Returns:
        True on success, False on failure.
    """
    try:
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.error("Failed to load image: %s", image_path)
            return False

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        if grayscale:
            result_array = edges  # Already single-channel
            mode = "L"
        else:
            rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            result_array = rgb_edges
            mode = "RGB"

        # Ensure parent directory exists before writing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(result_array, mode=mode).save(str(output_path))
        logger.info("Saved: %s", output_path)
        return True
    except Exception as exc:
        logger.error("Failed to process %s: %s", image_path, exc)
        return False


# ---------------------------------------------------------------------------
# 6. Input / Output helpers
# ---------------------------------------------------------------------------


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
                out = out_raw / f"{inp.stem}-edges.{fmt}"
            else:
                out = out_raw.with_suffix(f".{fmt}")
        else:
            out = inp.parent / f"{inp.stem}-edges.{fmt}"
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
        out_dir = input_paths[0].parent / "edges"

    return [out_dir / f"{p.stem}-edges.{fmt}" for p in input_paths]


# ---------------------------------------------------------------------------
# 7. Interactive mode
# ---------------------------------------------------------------------------


def interactive_configure(args: argparse.Namespace) -> argparse.Namespace:
    """
    Guide the user through configuration in the terminal.

    Overrides any command-line settings with the user's interactive choices.
    """
    print("\n" + "=" * 60)
    print("  Interactive Configuration Mode")
    print("=" * 60 + "\n")

    # --- Low threshold ---
    print("Set the Canny LOW threshold (0-100).")
    print("  Lower values detect more edges; higher values are stricter.")
    while True:
        choice = input("Enter value [0-100, default: 20]: ").strip()
        if not choice:
            args.low = 20
            break
        try:
            val = int(choice)
            if 0 <= val <= 100:
                args.low = val
                break
        except ValueError:
            pass
        print("  Invalid choice. Please enter a number between 0 and 100.")

    # --- High threshold ---
    print("\nSet the Canny HIGH threshold (0-100).")
    print("  Should typically be 2-3x the low threshold.")
    while True:
        choice = input("Enter value [0-100, default: 60]: ").strip()
        if not choice:
            args.high = 60
            break
        try:
            val = int(choice)
            if 0 <= val <= 100:
                args.high = val
                break
        except ValueError:
            pass
        print("  Invalid choice. Please enter a number between 0 and 100.")

    # --- Additional settings? ---
    if _ask_user("\nDo you want to specify additional settings? (output format)", default=False):
        # Format
        formats = ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]
        print("\nWhich output format do you want to use?")
        for i, fmt in enumerate(formats, 1):
            print(f"  {i}. {fmt}")
        while True:
            choice = input("Enter choice [1-6, default: 1 (png)]: ").strip()
            if not choice:
                args.format = formats[0]
                break
            try:
                idx = int(choice)
                if 1 <= idx <= len(formats):
                    args.format = formats[idx - 1]
                    break
            except ValueError:
                pass
            print("  Invalid choice. Please enter 1-6.")

    print("\n" + "=" * 60)
    print("  Configuration complete!")
    print("=" * 60 + "\n")
    return args


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments, optionally enter interactive mode, and process images."""
    parser = _build_parser()
    args = parser.parse_args()

    # Interactive configuration overrides CLI flags
    if args.interactive:
        args = interactive_configure(args)

    # Clamp thresholds to valid range
    low = max(0, min(100, args.low))
    high = max(0, min(100, args.high))
    if low != args.low or high != args.high:
        logger.warning(
            "Thresholds clamped to 0-100 range: low=%d, high=%d", low, high
        )

    # Map user-friendly 0-100 to OpenCV 0-255
    opencv_low = int(low * _THRESHOLD_SCALE)
    opencv_high = int(high * _THRESHOLD_SCALE)

    # Resolve and validate input
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    images = collect_images(input_path)
    if not images:
        logger.error("No valid images found to process.")
        return 1

    output_paths = resolve_output_paths(images, args.output, args.format)

    # Summary before processing
    logger.info("Input:  %s", input_path)
    logger.info("Images: %d", len(images))
    logger.info("Low:    %d (OpenCV: %d)", low, opencv_low)
    logger.info("High:   %d (OpenCV: %d)", high, opencv_high)
    logger.info("Format: %s", args.format)

    # Run processing
    success = 0
    for img_in, img_out in zip(images, output_paths):
        if detect_edges(
            img_in,
            img_out,
            opencv_low,
            opencv_high,
            grayscale=args.grayscale,
        ):
            success += 1

    logger.info(
        "Finished. Successfully processed %d / %d images.",
        success,
        len(images),
    )
    return 0 if success == len(images) else 1


if __name__ == "__main__":
    sys.exit(main())
