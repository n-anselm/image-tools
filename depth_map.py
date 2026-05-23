#!/usr/bin/env python3
"""
Created by OpenCode
Initialized: 2026-05-18
Updated: 2026-05-18

Depth Map Extraction CLI using MiDaS (DPT) via Transformers.

Generates depth maps from images or folders of images using MiDaS DPT models.
Output can be grayscale or various colormaps. By default, depth is inverted
for ControlNet compatibility.

Usage:
    python depth_map.py input.png [output.png] [options]
    python depth_map.py input_folder/ [output_folder/] [options]

Interactive Mode:
    python depth_map.py input.png -i

Environment:
    The script checks for a virtual environment on startup. If none is detected,
    it asks to create one. Missing dependencies (torch, transformers, opencv,
    Pillow, numpy) are also checked and can be auto-installed with confirmation.

Examples:
    # Basic usage: generates input-depthmap.png
    python depth_map.py photo.png

    # Specify output file
    python depth_map.py photo.png my_depth.png

    # Process a folder (creates ./photos/depthmap/ automatically)
    python depth_map.py ./photos/

    # Use a specific model and colormap
    python depth_map.py photo.png --model dpt-large --colormap viridis

    # Interactive mode to choose settings
    python depth_map.py photo.png -i
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
logger = logging.getLogger("depth_map")

# Supported image extensions for input scanning
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
    "torch": "torch",
    "transformers": "transformers",
}

# Approximate VRAM requirements per model (bytes)
_MODEL_VRAM_BYTES: dict[str, int] = {
    "Intel/dpt-hybrid-midas": 2 * 1024**3,
    "Intel/dpt-large": 4 * 1024**3,
}

# Pipeline cache: key="model_device", value=(pipeline, device_description)
_pipeline_cache: dict[str, tuple] = {}


# ---------------------------------------------------------------------------
# 2. CLI argument parsing (--help works before any heavy setup)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser with full help text."""
    parser = argparse.ArgumentParser(
        prog="depth_map.py",
        description="Generate depth maps from images using MiDaS DPT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive mode (-i) lets you configure settings step-by-step in the terminal.

Output rules:
  • Single file input + no output  ->  <name>-depthmap.<ext> in same folder
  • Single file input + output     ->  written to specified path
  • Folder input    + no output    ->  creates <input>/depthmap/ folder
  • Folder input    + output       ->  writes to specified output folder

Examples:
  %(prog)s photo.png
  %(prog)s photo.png depth.png --model dpt-large
  %(prog)s ./photos/ ./depthmaps/ --colormap viridis
  %(prog)s photo.png --close 10 --far 10 --gamma 0.8
  %(prog)s photo.png --auto-contrast --gamma 1.2
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

    # Core settings
    parser.add_argument(
        "--model",
        choices=["dpt-hybrid-midas", "dpt-large"],
        default="dpt-hybrid-midas",
        help="Model to use for depth estimation (default: dpt-hybrid-midas)",
    )
    parser.add_argument(
        "--colormap",
        choices=[
            "grayscale",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "jet",
            "turbo",
            "hot",
            "cool",
        ],
        default="grayscale",
        help="Output colormap (default: grayscale)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=(
            "Device for inference: auto picks GPU if enough VRAM is available; "
            "cuda forces GPU; cpu forces CPU (default: auto)"
        ),
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Do not invert depth values (by default, depth is inverted for ControlNet compatibility)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        default="png",
        help="Output image format for auto-generated paths (default: png)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for processing multiple images at once (default: 1)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for images when a folder is provided",
    )

    # Depth post-processing
    parser.add_argument(
        "--close",
        type=int,
        default=0,
        metavar="0-100",
        help="Percentile to clip from the near end. 10 = ignore the closest 10%% of pixels. (default: 0)",
    )
    parser.add_argument(
        "--far",
        type=int,
        default=0,
        metavar="0-100",
        help="Percentile to clip from the far end. 10 = ignore the farthest 10%% of pixels. (default: 0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        metavar="0.5-2.0",
        help="Gamma correction after stretching. < 1.0 brightens near-field; > 1.0 emphasizes distance. (default: 1.0)",
    )
    parser.add_argument(
        "--auto-contrast",
        action="store_true",
        help="Automatically compute --close and --far from the depth histogram (5th/95th percentile). Overrides explicit values.",
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
            # Non-interactive environment (e.g., piped input)
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
    logger.info("These packages are needed to run depth estimation.")
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
import torch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from transformers import pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# 5. Core depth-estimation logic (refactored from original GUI plugin)
# ---------------------------------------------------------------------------


def _device_label() -> str:
    """Return a human-readable string describing the available compute device."""
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"


def _choose_device(model_name: str, device_pref: str = "auto") -> tuple[int, str]:
    """
    Pick the safest device (GPU or CPU) based on available VRAM.

    Args:
        model_name: Hugging Face model identifier.
        device_pref: 'auto', 'cuda', or 'cpu'.

    Returns:
        (device_index, description_string)
    """
    if device_pref == "cpu":
        return -1, "CPU (user requested)"

    if not torch.cuda.is_available():
        if device_pref == "cuda":
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        return -1, "CPU"

    if device_pref == "cuda":
        return 0, f"CUDA ({torch.cuda.get_device_name(0)})"

    # Auto mode: check available VRAM before committing to GPU
    try:
        free_bytes, _total = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024**3)
        required = _MODEL_VRAM_BYTES.get(model_name, 2 * 1024**3)
        req_gb = required / (1024**3)
        logger.info("GPU 0: %.1f GB free. Model needs ~%.1f GB.", free_gb, req_gb)
        if free_bytes >= required:
            return 0, f"CUDA ({torch.cuda.get_device_name(0)})"
        logger.warning(
            "Insufficient VRAM (%.1f GB free, needs ~%.1f GB). Using CPU.",
            free_gb,
            req_gb,
        )
        return -1, "CPU (insufficient GPU VRAM)"
    except Exception as exc:
        logger.warning("GPU query failed (%s). Using CPU.", exc)
        return -1, "CPU (GPU query failed)"


def _get_pipeline(model_name: str, device_pref: str = "auto"):
    """
    Get or create a cached depth-estimation pipeline.

    Tries online download first; falls back to local cache if offline.
    """
    cache_key = f"{model_name}_{device_pref}"
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    device, device_desc = _choose_device(model_name, device_pref)
    logger.info("Loading depth model: %s on %s", model_name, device_desc)

    try:
        pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=device,
            local_files_only=False,
        )
        logger.info("Loaded %s (online)", model_name)
    except Exception as exc:
        logger.info("Online load failed (%s), using local cache...", exc)
        pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=device,
            local_files_only=True,
        )
        logger.info("Loaded %s from local cache", model_name)

    _pipeline_cache[cache_key] = (pipe, device_desc)
    return pipe, device_desc


def _colormap_from_name(name: str) -> int:
    """Map friendly colormap name to OpenCV colormap constant."""
    cmap_map = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
    }
    return cmap_map.get(name, cv2.COLORMAP_VIRIDIS)


def _normalize_depth(
    depth_np: np.ndarray,
    close: int,
    far: int,
    gamma: float,
    auto_contrast: bool,
) -> np.ndarray:
    """
    Normalize a raw depth map with optional clipping, stretching, and gamma.

    Pipeline:
      1. Clip percentile tails (close from near end, far from far end).
      2. Linearly stretch the remaining range to [0, 1].
      3. Apply gamma correction.
      4. Scale to [0, 255] and convert to uint8.

    Args:
        depth_np: Raw depth values as float32 array.
        close: Percentile to clip from the near (high) end (0-100).
        far: Percentile to clip from the far (low) end (0-100).
        gamma: Gamma exponent. 1.0 = linear.
        auto_contrast: If True, compute close/far from 5th/95th percentiles.

    Returns:
        Normalized depth as uint8 array in [0, 255].
    """
    if auto_contrast:
        p_low = float(np.percentile(depth_np, 5.0))
        p_high = float(np.percentile(depth_np, 95.0))
        logger.info(
            "Auto-contrast: clipping at raw depth values far=%.2f, close=%.2f",
            p_low,
            p_high,
        )
    else:
        p_low = float(np.percentile(depth_np, float(far)))
        p_high = float(np.percentile(depth_np, 100.0 - float(close)))

    if p_high <= p_low:
        logger.warning(
            "Depth clip range is empty (low=%.4f, high=%.4f). Falling back to min-max.",
            p_low,
            p_high,
        )
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_norm = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_np)
    else:
        clipped = np.clip(depth_np, p_low, p_high)
        depth_norm = (clipped - p_low) / (p_high - p_low)

    # Gamma: applied to the already-stretched [0,1] range
    if gamma != 1.0:
        depth_norm = np.power(depth_norm, float(gamma))

    return (255 * depth_norm).astype(np.uint8)


# ---------------------------------------------------------------------------
# 6. Input / Output helpers
# ---------------------------------------------------------------------------


def collect_images(input_path: Path, recursive: bool = False) -> list[Path]:
    """
    Collect all supported image paths from a file or folder.

    Args:
        input_path: File or directory to scan.
        recursive: Whether to recurse into subdirectories.

    Returns:
        Sorted list of unique image Paths.
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
    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            images.update(input_path.rglob(f"*{ext}"))
            images.update(input_path.rglob(f"*{ext.upper()}"))
    else:
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
                out = out_raw / f"{inp.stem}-depthmap.{fmt}"
            else:
                out = out_raw.with_suffix(f".{fmt}")
        else:
            out = inp.parent / f"{inp.stem}-depthmap.{fmt}"
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
        out_dir = input_paths[0].parent / "depthmap"

    return [out_dir / f"{p.stem}-depthmap.{fmt}" for p in input_paths]


# ---------------------------------------------------------------------------
# 7. Processing engine
# ---------------------------------------------------------------------------


def process_images(
    image_paths: list[Path],
    output_paths: list[Path],
    model_name: str,
    colormap_name: str,
    device_pref: str,
    invert: bool,
    batch_size: int,
    close: int,
    far: int,
    gamma: float,
    auto_contrast: bool,
) -> int:
    """
    Run depth estimation on a list of images and save the results.

    Args:
        image_paths: Input images.
        output_paths: Destination paths (same length as image_paths).
        model_name: Full Hugging Face model identifier (e.g., Intel/dpt-large).
        colormap_name: 'grayscale' or an OpenCV colormap name.
        device_pref: 'auto', 'cuda', or 'cpu'.
        invert: Whether to invert depth (ControlNet style).
        batch_size: How many images to feed the pipeline at once.
        close: Percentile to clip from the near end.
        far: Percentile to clip from the far end.
        gamma: Gamma correction exponent.
        auto_contrast: Whether to auto-compute clip percentiles.

    Returns:
        Number of successfully processed images.
    """
    pipe, device_desc = _get_pipeline(model_name, device_pref)
    logger.info("Inference device: %s", device_desc)

    use_grayscale = colormap_name.lower() == "grayscale"
    cmap = _colormap_from_name(colormap_name) if not use_grayscale else None

    total = len(image_paths)
    success_count = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_in = image_paths[batch_start:batch_end]
        batch_out = output_paths[batch_start:batch_end]

        # Load images into PIL format for the transformers pipeline
        pil_images: list[PILImage.Image] = []
        valid_out_paths: list[Path] = []
        for img_path, out_path in zip(batch_in, batch_out):
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.warning("Failed to load image, skipping: %s", img_path)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(PILImage.fromarray(img_rgb))
            valid_out_paths.append(out_path)

        if not pil_images:
            continue

        # Run inference -------------------------------------------------------
        try:
            results = pipe(pil_images)
            if not isinstance(results, list):
                results = [results]
        except Exception as exc:
            logger.error("Batch inference failed: %s", exc)
            # Fallback: process one-by-one to isolate failures
            results = []
            for pil_img in pil_images:
                try:
                    results.append(pipe(pil_img))
                except Exception as exc2:
                    logger.error("Single image inference failed: %s", exc2)
                    results.append(None)

        # Save results --------------------------------------------------------
        for result, out_path in zip(results, valid_out_paths):
            if result is None:
                continue

            depth = result["depth"]  # PIL Image, mode=L
            depth_np = np.array(depth).astype(np.float32)

            # Normalize with optional clipping, stretching, and gamma
            depth_8bit = _normalize_depth(
                depth_np, close, far, gamma, auto_contrast
            )

            # Invert for ControlNet convention by default
            if invert:
                depth_8bit = 255 - depth_8bit

            if use_grayscale:
                result_array = depth_8bit
                mode = "L"
            else:
                depth_colored = cv2.applyColorMap(depth_8bit, cmap)
                result_array = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
                mode = "RGB"

            # Ensure parent directory exists before writing
            out_path.parent.mkdir(parents=True, exist_ok=True)

            PILImage.fromarray(result_array, mode=mode).save(str(out_path))
            logger.info("Saved: %s", out_path)
            success_count += 1

        logger.info("Progress: %d / %d", batch_end, total)

    logger.info(
        "Finished. Successfully processed %d / %d images.",
        success_count,
        total,
    )
    return success_count


# ---------------------------------------------------------------------------
# 8. Interactive mode
# ---------------------------------------------------------------------------


def interactive_configure(args: argparse.Namespace) -> argparse.Namespace:
    """
    Guide the user through configuration in the terminal.

    Overrides any command-line settings with the user's interactive choices.
    """
    print("\n" + "=" * 60)
    print("  Interactive Configuration Mode")
    print("=" * 60 + "\n")

    # --- Model selection ---
    models = ["dpt-hybrid-midas", "dpt-large"]
    print("Which model do you want to use?")
    print("  1. dpt-hybrid-midas  (faster, ~2 GB VRAM)")
    print("  2. dpt-large         (better quality, ~4 GB VRAM)")
    while True:
        choice = input("Enter choice [1-2, default: 1]: ").strip()
        if not choice:
            args.model = models[0]
            break
        if choice == "1":
            args.model = models[0]
            break
        if choice == "2":
            args.model = models[1]
            break
        print("  Invalid choice. Please enter 1 or 2.")

    # --- Colormap selection ---
    colormaps = [
        "grayscale",
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "jet",
        "turbo",
        "hot",
        "cool",
    ]
    print("\nWhich colormap do you want to use?")
    for i, cmap in enumerate(colormaps, 1):
        print(f"  {i}. {cmap}")
    while True:
        choice = input(
            f"Enter choice [1-{len(colormaps)}, default: 1 (grayscale)]: "
        ).strip()
        if not choice:
            args.colormap = colormaps[0]
            break
        try:
            idx = int(choice)
            if 1 <= idx <= len(colormaps):
                args.colormap = colormaps[idx - 1]
                break
        except ValueError:
            pass
        print(f"  Invalid choice. Please enter 1-{len(colormaps)}.")

    # --- Additional settings? ---
    if _ask_user("\nDo you want to specify additional settings? (CUDA/CPU, depth inversion, output format, batch size)", default=False):
        # Device
        devices = ["auto", "cuda", "cpu"]
        print("\nWhich device do you want to use?")
        for i, dev in enumerate(devices, 1):
            print(f"  {i}. {dev}")
        while True:
            choice = input("Enter choice [1-3, default: 1 (auto)]: ").strip()
            if not choice:
                args.device = devices[0]
                break
            try:
                idx = int(choice)
                if 1 <= idx <= len(devices):
                    args.device = devices[idx - 1]
                    break
            except ValueError:
                pass
            print("  Invalid choice. Please enter 1-3.")

        # Invert
        args.no_invert = _ask_user(
            "Disable depth inversion (for ControlNet compatibility)?",
            default=False,
        )

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

        # Batch size
        while True:
            choice = input("Enter batch size [default: 1]: ").strip()
            if not choice:
                args.batch = 1
                break
            try:
                batch = int(choice)
                if batch >= 1:
                    args.batch = batch
                    break
            except ValueError:
                pass
            print("  Invalid choice. Please enter a positive integer.")

        # Depth post-processing
        print("\n--- Depth Post-Processing ---")
        args.auto_contrast = _ask_user(
            "Use auto-contrast (automatically clip depth tails)?",
            default=False,
        )
        if not args.auto_contrast:
            # Close clip
            while True:
                choice = input(
                    "Clip close percentile [0-100, default: 0]: "
                ).strip()
                if not choice:
                    args.close = 0
                    break
                try:
                    val = int(choice)
                    if 0 <= val <= 100:
                        args.close = val
                        break
                except ValueError:
                    pass
                print("  Invalid choice. Please enter 0-100.")

            # Far clip
            while True:
                choice = input(
                    "Clip far percentile [0-100, default: 0]: "
                ).strip()
                if not choice:
                    args.far = 0
                    break
                try:
                    val = int(choice)
                    if 0 <= val <= 100:
                        args.far = val
                        break
                except ValueError:
                    pass
                print("  Invalid choice. Please enter 0-100.")

        # Gamma
        while True:
            choice = input(
                "Gamma correction [0.1-3.0, default: 1.0]: "
            ).strip()
            if not choice:
                args.gamma = 1.0
                break
            try:
                val = float(choice)
                if 0.1 <= val <= 3.0:
                    args.gamma = val
                    break
            except ValueError:
                pass
            print("  Invalid choice. Please enter 0.1-3.0.")

    print("\n" + "=" * 60)
    print("  Configuration complete!")
    print("=" * 60 + "\n")
    return args


# ---------------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments, optionally enter interactive mode, and process images."""
    parser = _build_parser()
    args = parser.parse_args()

    # Interactive configuration overrides CLI flags
    if args.interactive:
        args = interactive_configure(args)

    # Validate depth clip ranges
    if not args.auto_contrast and args.close + args.far >= 100:
        logger.error(
            "Invalid clip range: --close %d + --far %d = %d (must be < 100).",
            args.close,
            args.far,
            args.close + args.far,
        )
        return 1

    # Resolve and validate input
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    images = collect_images(input_path, recursive=args.recursive)
    if not images:
        logger.error("No valid images found to process.")
        return 1

    output_paths = resolve_output_paths(images, args.output, args.format)

    # Summary before processing
    logger.info("Input:    %s", input_path)
    logger.info("Images:   %d", len(images))
    logger.info("Model:    Intel/%s", args.model)
    logger.info("Colormap: %s", args.colormap)
    logger.info("Invert:   %s", not args.no_invert)
    logger.info("Device:   %s", args.device)
    logger.info("Format:   %s", args.format)
    logger.info("Batch:    %d", args.batch)
    if args.auto_contrast:
        logger.info("Contrast: auto")
    else:
        logger.info("Contrast: close=%d far=%d gamma=%.2f", args.close, args.far, args.gamma)

    # Run processing
    try:
        success = process_images(
            image_paths=images,
            output_paths=output_paths,
            model_name=f"Intel/{args.model}",
            colormap_name=args.colormap,
            device_pref=args.device,
            invert=not args.no_invert,
            batch_size=args.batch,
            close=args.close,
            far=args.far,
            gamma=args.gamma,
            auto_contrast=args.auto_contrast,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130

    return 0 if success == len(images) else 1


if __name__ == "__main__":
    sys.exit(main())
