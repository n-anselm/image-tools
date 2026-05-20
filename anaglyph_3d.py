#!/usr/bin/env python3
"""
Created by OpenCode.
Initialized: 2026-05-20
Updated: 2026-05-20

Anaglyph 3D Image Generator CLI.

Creates red-cyan anaglyph stereo images from an RGB photo and its depth map.
The red channel is shifted horizontally based on depth values to produce a 3D
effect viewable with red-cyan glasses. Near objects shift more than far objects,
creating the illusion of depth.

By default, the script automatically looks for a depth map named
<stem>-depthmap.png next to the input image. If it is not found, the script
can optionally run depth_map.py to generate one.

Usage:
    python anaglyph_3d.py photo.png [options]
    python anaglyph_3d.py photo.png -d photo-depthmap.png [options]
    python anaglyph_3d.py photo.png -i

Examples:
    # Auto-look for photo-depthmap.png and create photo-anaglyph.png
    python anaglyph_3d.py photo.png

    # Specify depth map explicitly
    python anaglyph_3d.py photo.png --depth photo-depthmap.png

    # Adjust parallax strength
    python anaglyph_3d.py photo.png --strength 1.5

    # Interactive configuration
    python anaglyph_3d.py photo.png -i
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
logger = logging.getLogger("anaglyph_3d")

SUPPORTED_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}

REQUIRED_PACKAGES: dict[str, str] = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "Pillow",
}

# ---------------------------------------------------------------------------
# 2. CLI argument parsing (--help works before any heavy setup)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser with full help text."""
    parser = argparse.ArgumentParser(
        prog="anaglyph_3d.py",
        description="Generate red-cyan anaglyph 3D images from photos and depth maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive mode (-i) lets you configure settings step-by-step in the terminal.

Depth map resolution:
  • Not specified          -> auto-look for <stem>-depthmap.png next to input
  • --depth <path>         -> use the provided depth map image
  • Auto-lookup missing    -> prompt to run depth_map.py to generate one

Output rules:
  • Single file + no output  ->  <name>-anaglyph.<ext> in same folder
  • Single file + output      ->  written to specified path

Examples:
  %(prog)s photo.png
  %(prog)s photo.png --strength 1.5
  %(prog)s photo.png --depth photo-depthmap.png out.png
  %(prog)s photo.png -i
        """,
    )

    parser.add_argument(
        "input",
        help="Input RGB image file",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output image file (optional)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enter interactive mode to configure settings step-by-step",
    )

    # Core settings
    parser.add_argument(
        "--depth",
        dest="depth_input",
        metavar="PATH",
        help="Explicit path to a depth map image (overrides auto-lookup)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        metavar="0.0-2.0",
        help="Parallax strength multiplier. 0.0 = no effect, 1.0 = default, 2.0 = strong (default: 1.0)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        default="png",
        help="Output image format for auto-generated paths (default: png)",
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
            logger.info("Non-interactive environment detected. Using default: %s", default)
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
    """Ensure required packages are installed."""
    missing = _check_dependencies()
    if not missing:
        return

    logger.warning("Missing required packages: %s", ", ".join(missing))
    logger.info("These packages are needed to run anaglyph generation.")
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

if "--help" in sys.argv or "-h" in sys.argv:
    _build_parser().print_help()
    sys.exit(0)

ensure_environment()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from PIL import ImageOps  # noqa: E402


# ---------------------------------------------------------------------------
# 5. Core anaglyph logic
# ---------------------------------------------------------------------------


def _find_depth_map_script() -> Path | None:
    """Return the path to depth_map.py in the same directory as this script."""
    script_dir = Path(__file__).resolve().parent
    depth_script = script_dir / "depth_map.py"
    if depth_script.exists():
        return depth_script
    logger.warning("depth_map.py not found in script directory: %s", script_dir)
    return None


def _run_depth_map(rgb_path: Path) -> Path | None:
    """
    Run depth_map.py to generate a depth map for the given image.

    Args:
        rgb_path: Path to the RGB image.

    Returns:
        Path to the generated depth map, or None if generation failed.
    """
    depth_script = _find_depth_map_script()
    if depth_script is None:
        return None

    logger.info("Running depth_map.py to generate depth map...")
    result = subprocess.run(
        [sys.executable, str(depth_script), str(rgb_path)],
        cwd=str(depth_script.parent),
    )

    if result.returncode != 0:
        logger.error("depth_map.py failed with exit code %d", result.returncode)
        return None

    # depth_map.py default output: <stem>-depthmap.png
    expected = rgb_path.parent / f"{rgb_path.stem}-depthmap.png"
    if expected.exists():
        logger.info("Generated depth map: %s", expected)
        return expected

    # Fallback: search for any matching depthmap file
    candidates = list(rgb_path.parent.glob(f"{rgb_path.stem}-depthmap.*"))
    if candidates:
        logger.info("Generated depth map: %s", candidates[0])
        return candidates[0]

    logger.error("depth_map.py completed but no output file was found.")
    return None


def find_depth_map(rgb_path: Path, depth_arg: str | None) -> Path | None:
    """
    Resolve the depth map path.

    Resolution order:
      1. If --depth is provided, use it directly.
      2. Auto-look for <stem>-depthmap.png next to the RGB image.
      3. If not found, ask to run depth_map.py, then check again.

    Args:
        rgb_path: Path to the input RGB image.
        depth_arg: Optional explicit depth map path from --depth.

    Returns:
        Resolved Path to the depth map, or None if unavailable.
    """
    if depth_arg:
        depth_path = Path(depth_arg).resolve()
        if not depth_path.exists():
            logger.error("Specified depth map not found: %s", depth_path)
            return None
        logger.info("Using explicit depth map: %s", depth_path)
        return depth_path

    # Auto-lookup
    expected = rgb_path.parent / f"{rgb_path.stem}-depthmap.png"
    if expected.exists():
        logger.info("Found depth map: %s", expected)
        return expected

    # Not found — prompt user
    logger.warning("Depth map not found: %s", expected)
    if _ask_user("Run depth_map.py to generate the depth map?", default=True):
        generated = _run_depth_map(rgb_path)
        if generated is not None and generated.exists():
            return generated

    logger.error(
        "No depth map available. Provide one with --depth or generate it first."
    )
    return None


def load_image(path: Path) -> np.ndarray | None:
    """
    Load an image from disk, preserving EXIF orientation.

    PIL is used for loading so that EXIF Orientation metadata is respected,
    then the image is converted to an OpenCV-compatible BGR numpy array.

    Args:
        path: Path to the image file.

    Returns:
        Loaded image as a numpy array (BGR for color, grayscale for single-channel),
        or None if loading failed.
    """
    try:
        pil_img = PILImage.open(str(path))
        pil_img = ImageOps.exif_transpose(pil_img)
        img_rgb = np.array(pil_img)

        if len(img_rgb.shape) == 3 and img_rgb.shape[2] >= 3:
            # Convert RGB(A) → BGR(A)
            if img_rgb.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGRA)
        else:
            img_bgr = img_rgb

        logger.debug("Loaded image: %s (%s)", path, img_bgr.shape)
        return img_bgr
    except Exception as exc:
        logger.error("Failed to load image: %s — %s", path, exc)
        return None


def create_anaglyph(rgb: np.ndarray, depth: np.ndarray, strength: float) -> np.ndarray:
    """
    Create a red-cyan anaglyph from an RGB image and a depth map.

    The red channel is shifted left and the green/blue channels are shifted right
    based on depth values. High depth values (near objects) produce larger shifts.

    Args:
        rgb: Input RGB image in BGR format (OpenCV default), shape (H, W, 3).
        depth: Depth map image. Single-channel or 3-channel (auto-converted to grayscale).
        strength: Parallax strength multiplier. 0.0 = no effect, 1.0 = default,
                 higher values produce stronger 3D effect.

    Returns:
        Anaglyph image in BGR format, same shape as rgb.
    """
    h, w = rgb.shape[:2]

    # Ensure depth is single-channel grayscale
    if len(depth.shape) == 3:
        depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    else:
        depth_gray = depth

    # Normalize depth to [0, 1]
    depth_norm = depth_gray.astype(np.float32) / 255.0

    # Compute per-pixel horizontal shift
    # Max shift is 3% of image width at full strength
    max_shift = w * 0.03
    shifts = (depth_norm * max_shift * strength).astype(np.int32)

    # Split BGR channels
    b, g, r = cv2.split(rgb)

    # Create coordinate grids
    y_coords, x_coords = np.indices((h, w))

    # Shift red channel LEFT (negative x direction)
    red_target_x = x_coords - shifts
    red_valid = (red_target_x >= 0) & (red_target_x < w)
    r_shifted = np.zeros_like(r)
    r_shifted[y_coords[red_valid], red_target_x[red_valid]] = r[
        y_coords[red_valid], x_coords[red_valid]
    ]

    # Shift green and blue channels RIGHT (positive x direction)
    gb_target_x = x_coords + shifts
    gb_valid = (gb_target_x >= 0) & (gb_target_x < w)
    g_shifted = np.zeros_like(g)
    b_shifted = np.zeros_like(b)
    g_shifted[y_coords[gb_valid], gb_target_x[gb_valid]] = g[
        y_coords[gb_valid], x_coords[gb_valid]
    ]
    b_shifted[y_coords[gb_valid], gb_target_x[gb_valid]] = b[
        y_coords[gb_valid], x_coords[gb_valid]
    ]

    # Merge back to BGR
    anaglyph = cv2.merge([b_shifted, g_shifted, r_shifted])
    return anaglyph


def resolve_output_path(rgb_path: Path, output_arg: str | None, fmt: str) -> Path:
    """
    Determine the output file path.

    Args:
        rgb_path: Path to the input RGB image.
        output_arg: Optional user-provided output path.
        fmt: Output file extension (without dot).

    Returns:
        Resolved output Path.
    """
    fmt = fmt.lower().lstrip(".")

    if output_arg:
        out = Path(output_arg)
        # If user ended with a path separator or the path is an existing directory,
        # treat it as a folder and append a default filename.
        if output_arg.endswith(("/", "\\")) or (out.exists() and out.is_dir()):
            return out / f"{rgb_path.stem}-anaglyph.{fmt}"
        return out.with_suffix(f".{fmt}")

    return rgb_path.parent / f"{rgb_path.stem}-anaglyph.{fmt}"


# ---------------------------------------------------------------------------
# 6. Interactive mode
# ---------------------------------------------------------------------------


def interactive_configure(args: argparse.Namespace) -> argparse.Namespace:
    """
    Guide the user through configuration in the terminal.

    Overrides any command-line settings with the user's interactive choices.
    """
    print("\n" + "=" * 60)
    print("  Interactive Configuration Mode")
    print("=" * 60 + "\n")

    # --- Strength selection ---
    print("Parallax strength controls how strongly pixels shift based on depth.")
    print("  0.0 = flat (no 3D effect)")
    print("  1.0 = default (good balance)")
    print("  2.0 = strong (more dramatic depth, may show edge gaps)")
    while True:
        choice = input(
            f"Enter strength [0.0-2.0, default: {args.strength}]: "
        ).strip()
        if not choice:
            break
        try:
            val = float(choice)
            if 0.0 <= val <= 2.0:
                args.strength = val
                break
            print("  Please enter a value between 0.0 and 2.0.")
        except ValueError:
            print("  Invalid number.")

    # --- Additional settings? ---
    if _ask_user("Do you want to specify additional settings?", default=False):
        # Depth map
        depth_input = input(
            "Depth map path (leave empty for auto-lookup): "
        ).strip()
        if depth_input:
            args.depth_input = depth_input

        # Output format
        formats = ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]
        print("\nOutput format:")
        for i, fmt in enumerate(formats, 1):
            print(f"  {i}. {fmt}")
        while True:
            choice = input(f"Enter choice [1-{len(formats)}, default: 1]: ").strip()
            if not choice:
                break
            try:
                idx = int(choice)
                if 1 <= idx <= len(formats):
                    args.format = formats[idx - 1]
                    break
            except ValueError:
                pass
            print(f"  Invalid choice. Please enter 1-{len(formats)}.")

    print("\n" + "=" * 60)
    print("  Configuration complete!")
    print("=" * 60 + "\n")
    return args


# ---------------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments, optionally enter interactive mode, and generate anaglyph."""
    parser = _build_parser()
    args = parser.parse_args()

    # Interactive configuration overrides CLI flags
    if args.interactive:
        args = interactive_configure(args)

    # Resolve and validate input
    rgb_path = Path(args.input).resolve()
    if not rgb_path.exists():
        logger.error("Input image not found: %s", rgb_path)
        return 1

    if not rgb_path.is_file():
        logger.error(
            "Input must be a single image file. Folder batch mode is not yet supported: %s",
            rgb_path,
        )
        return 1

    if rgb_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported input file format: %s", rgb_path.suffix)
        return 1

    # Find or generate depth map
    depth_path = find_depth_map(rgb_path, args.depth_input)
    if depth_path is None:
        return 1

    # Resolve output path
    output_path = resolve_output_path(rgb_path, args.output, args.format)

    # Load images
    rgb = load_image(rgb_path)
    depth = load_image(depth_path)
    if rgb is None or depth is None:
        return 1

    # Validate dimensions match; resize depth if needed
    if rgb.shape[:2] != depth.shape[:2]:
        logger.warning(
            "RGB (%s) and depth map (%s) have different dimensions. "
            "Resizing depth map to match RGB.",
            rgb.shape[:2],
            depth.shape[:2],
        )
        depth = cv2.resize(
            depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    # Summary before processing
    logger.info("Input:    %s", rgb_path)
    logger.info("Depth:    %s", depth_path)
    logger.info("Output:   %s", output_path)
    logger.info("Strength: %.2f", args.strength)
    logger.info("Format:   %s", args.format)

    # Generate anaglyph
    try:
        anaglyph = create_anaglyph(rgb, depth, args.strength)
    except Exception as exc:
        logger.error("Failed to create anaglyph: %s", exc)
        return 1

    # Save
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), anaglyph)
        logger.info("Saved anaglyph: %s", output_path)
    except Exception as exc:
        logger.error("Failed to save output: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
