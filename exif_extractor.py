#!/usr/bin/env python3
"""
Created by OpenCode.
Initialized: 2026-05-20
Updated: 2026-05-20

EXIF Metadata Extractor CLI.

Extracts camera metadata (focal length, sensor size, GPS, exposure, etc.) from
image files or entire folders. Outputs structured JSON for machine readability
and optionally a human-readable Markdown summary in interactive mode.

Computed camera intrinsics (focal length in pixels, sensor width) are included
when sufficient EXIF tags are present, making this a useful preprocessing step
for photogrammetry and point-cloud generation.

Usage:
    python exif_extractor.py photo.jpg [output.json] [options]
    python exif_extractor.py ./photos/ [output.json] [options]

Interactive Mode:
    python exif_extractor.py photo.jpg -i

Environment:
    The script checks for a virtual environment on startup. If none is detected,
    it asks to create one. If a .venv already exists in the script's directory,
    the script restarts into it automatically. Missing dependencies (Pillow) are
checked and can be auto-installed with confirmation.

Examples:
    # Extract EXIF from a single image -> photo-exif.json
    python exif_extractor.py photo.jpg

    # Extract EXIF from a folder -> photos-exif-batch.json
    python exif_extractor.py ./photos/

    # Specify output file
    python exif_extractor.py photo.jpg metadata.json

    # Interactive mode (includes optional Markdown export)
    python exif_extractor.py photo.jpg -i
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. Light-weight setup (no heavy imports yet)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exif_extractor")

SUPPORTED_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
}

REQUIRED_PACKAGES: dict[str, str] = {
    "PIL": "Pillow",
}

# ---------------------------------------------------------------------------
# 2. CLI argument parsing (--help works before any heavy setup)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser with full help text."""
    parser = argparse.ArgumentParser(
        prog="exif_extractor.py",
        description="Extract EXIF metadata from images into structured JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive mode (-i) lets you configure settings step-by-step in the terminal.

Output rules:
  • Single file + no output  ->  <name>-exif.json in same folder
  • Single file + output     ->  written to specified path
  • Folder + no output       ->  <input>/EXIF/exif-batch.json
  • Folder + output          ->  written to specified path

Examples:
  %(prog)s photo.jpg
  %(prog)s ./photos/
  %(prog)s photo.jpg metadata.json
  %(prog)s photo.jpg -i
        """,
    )

    parser.add_argument(
        "input",
        help="Input image file or folder containing images",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output JSON file (optional)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enter interactive mode to configure settings step-by-step",
    )

    parser.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format (default: json)",
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
    logger.info("These packages are needed to extract EXIF metadata.")
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

from PIL import ExifTags  # noqa: E402
from PIL import Image as PILImage  # noqa: E402


# ---------------------------------------------------------------------------
# 5. Core EXIF extraction logic
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    """Convert an EXIF value (int, float, tuple, IFDRational) to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # IFDRational or tuple like (72, 1)
    if isinstance(value, tuple) and len(value) == 2:
        if value[1] == 0:
            return None
        return float(value[0]) / float(value[1])
    # Some PIL versions use IFDRational objects
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _format_exposure_time(value: Any) -> str | None:
    """Format exposure time as a human-readable fraction."""
    f = _to_float(value)
    if f is None or f <= 0:
        return None
    if f >= 1.0:
        return f"{f:.2f}s"
    # Find a nice fraction
    from fractions import Fraction
    frac = Fraction(f).limit_denominator(8000)
    return f"1/{frac.denominator}" if frac.numerator == 1 else str(frac)


def _format_gps_coord(value: Any, ref: str | None) -> float | None:
    """Convert GPS coordinate tuple (deg, min, sec) to decimal degrees."""
    if value is None or ref is None:
        return None
    try:
        deg = _to_float(value[0])
        mins = _to_float(value[1])
        sec = _to_float(value[2])
        if deg is None or mins is None or sec is None:
            return None
        decimal = deg + mins / 60.0 + sec / 3600.0
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (IndexError, TypeError):
        return None


def _format_exposure_program(value: int | None) -> str | None:
    """Map ExposureProgram EXIF value to human-readable string."""
    if value is None:
        return None
    mapping = {
        0: "Not defined",
        1: "Manual",
        2: "Program AE",
        3: "Aperture-priority AE",
        4: "Shutter speed priority AE",
        5: "Creative (Slow speed)",
        6: "Action (High speed)",
        7: "Portrait mode",
        8: "Landscape mode",
    }
    return mapping.get(value, f"Unknown ({value})")


def extract_exif(image_path: Path) -> dict[str, Any] | None:
    """
    Extract EXIF metadata from a single image.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary of extracted metadata, or None if extraction failed.
    """
    try:
        with PILImage.open(image_path) as img:
            width, height = img.size
            data: dict[str, Any] = {
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }

            exif = img._getexif()
            if not exif:
                logger.warning("No EXIF data found in: %s", image_path.name)
                return data

            # Map numeric tags to friendly names
            tag_map = {num: name for num, name in ExifTags.TAGS.items()}
            gps_tag_map = {num: name for num, name in ExifTags.GPSTAGS.items()}

            raw: dict[str, Any] = {}
            gps_raw: dict[str, Any] = {}

            for tag_id, value in exif.items():
                tag_name = tag_map.get(tag_id, f"Tag{tag_id}")
                raw[tag_name] = value

            # GPS info is nested
            gps_info = raw.get("GPSInfo")
            if gps_info and isinstance(gps_info, dict):
                for tag_id, value in gps_info.items():
                    tag_name = gps_tag_map.get(tag_id, f"GPSTag{tag_id}")
                    gps_raw[tag_name] = value

            # --- Image basics ---
            data["orientation"] = raw.get("Orientation")

            # --- Camera ---
            data["camera_make"] = raw.get("Make")
            data["camera_model"] = raw.get("Model")

            # --- Lens ---
            data["focal_length_mm"] = _to_float(raw.get("FocalLength"))
            data["focal_length_35mm_equiv"] = _to_float(
                raw.get("FocalLengthIn35mmFilm")
            )
            data["f_number"] = _to_float(raw.get("FNumber"))
            data["aperture_value"] = _to_float(raw.get("ApertureValue"))
            data["max_aperture_value"] = _to_float(raw.get("MaxApertureValue"))

            # --- Sensor ---
            data["focal_plane_x_resolution"] = _to_float(
                raw.get("FocalPlaneXResolution")
            )
            data["focal_plane_y_resolution"] = _to_float(
                raw.get("FocalPlaneYResolution")
            )
            data["focal_plane_resolution_unit"] = raw.get("FocalPlaneResolutionUnit")

            # --- Exposure ---
            data["exposure_time"] = _format_exposure_time(raw.get("ExposureTime"))
            data["iso_speed"] = raw.get("ISOSpeedRatings")
            data["exposure_program"] = _format_exposure_program(
                raw.get("ExposureProgram")
            )
            data["exposure_bias"] = _to_float(raw.get("ExposureBiasValue"))

            # --- GPS ---
            lat = _format_gps_coord(
                gps_raw.get("GPSLatitude"), gps_raw.get("GPSLatitudeRef")
            )
            lon = _format_gps_coord(
                gps_raw.get("GPSLongitude"), gps_raw.get("GPSLongitudeRef")
            )
            alt = _to_float(gps_raw.get("GPSAltitude"))
            alt_ref = gps_raw.get("GPSAltitudeRef")
            if alt is not None and alt_ref == 1:
                alt = -alt

            data["gps_latitude"] = lat
            data["gps_longitude"] = lon
            data["gps_altitude_m"] = alt

            # --- Timestamp ---
            data["datetime_original"] = raw.get("DateTimeOriginal")
            data["datetime_digitized"] = raw.get("DateTimeDigitized")
            data["datetime"] = raw.get("DateTime")

            # --- Computed intrinsics (when possible) ---
            intrinsics: dict[str, Any] = {}
            focal_mm = data.get("focal_length_mm")
            fp_x_res = data.get("focal_plane_x_resolution")
            fp_unit = data.get("focal_plane_resolution_unit")

            if focal_mm and fp_x_res and fp_unit:
                # Unit: 2 = inch, 3 = cm
                unit_mm = 25.4 if fp_unit == 2 else 10.0 if fp_unit == 3 else None
                if unit_mm:
                    pixels_per_mm = fp_x_res / unit_mm
                    sensor_width_mm = width / pixels_per_mm
                    focal_length_px = focal_mm * width / sensor_width_mm

                    intrinsics["sensor_width_mm"] = round(sensor_width_mm, 2)
                    intrinsics["sensor_height_mm"] = round(
                        height / pixels_per_mm, 2
                    )
                    intrinsics["focal_length_px"] = round(focal_length_px, 2)
                    intrinsics["principal_point_x"] = width // 2
                    intrinsics["principal_point_y"] = height // 2

            if intrinsics:
                data["computed_intrinsics"] = intrinsics

            return data

    except Exception as exc:
        logger.error("Failed to extract EXIF from %s: %s", image_path.name, exc)
        return None


# ---------------------------------------------------------------------------
# 6. Input / Output helpers
# ---------------------------------------------------------------------------


def collect_images(input_path: Path) -> list[Path]:
    """
    Collect all supported image paths from a file or folder.

    Args:
        input_path: File or directory to scan.

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
    for ext in SUPPORTED_EXTENSIONS:
        images.update(input_path.glob(f"*{ext}"))
        images.update(input_path.glob(f"*{ext.upper()}"))

    if not images:
        logger.warning("No images found in: %s", input_path)

    return sorted(p.resolve() for p in images)


def resolve_output_path(
    input_path: Path,
    output_arg: str | None,
    is_batch: bool,
) -> Path:
    """
    Determine the output JSON file path.

    Args:
        input_path: The input file or folder path.
        output_arg: Optional user-provided output path.
        is_batch: Whether processing a folder (batch mode).

    Returns:
        Resolved output Path.
    """
    if output_arg:
        out = Path(output_arg)
        # If user explicitly passed a directory, place default filename inside it.
        if output_arg.endswith(("/", "\\")) or (out.exists() and out.is_dir()):
            if is_batch:
                return out / f"{input_path.name}-exif-batch.json"
            return out / f"{input_path.stem}-exif.json"
        # Ensure .json extension
        return out.with_suffix(".json")

    if is_batch:
        # Place inside the folder under EXIF/ subdirectory
        return input_path / "EXIF" / "exif-batch.json"

    return input_path.parent / f"{input_path.stem}-exif.json"


def write_json(data: Any, path: Path) -> bool:
    """Write data to a JSON file with pretty formatting."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved JSON: %s", path)
        return True
    except Exception as exc:
        logger.error("Failed to write JSON: %s", exc)
        return False


def _markdown_escape(value: Any) -> str:
    """Escape pipe characters for markdown tables."""
    text = str(value) if value is not None else "—"
    return text.replace("|", "\\|")


def write_markdown(data: dict[str, Any] | list[dict[str, Any]], path: Path) -> bool:
    """Write EXIF data to a human-readable Markdown file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = ["# EXIF Metadata\n"]

        if isinstance(data, dict) and "images" in data:
            # Batch mode
            lines.append(f"**Source:** {data.get('source', 'N/A')}\n")
            lines.append(f"**Image count:** {data.get('image_count', 0)}\n\n")
            for img in data["images"]:
                lines.append(f"## {img.get('file_name', 'Unknown')}\n")
                lines.append("| Field | Value |")
                lines.append("|-------|-------|")
                for key, value in sorted(img.items()):
                    if key == "computed_intrinsics" and isinstance(value, dict):
                        for ckey, cval in sorted(value.items()):
                            lines.append(
                                f"| `{ckey}` | {_markdown_escape(cval)} |"
                            )
                    else:
                        lines.append(f"| `{key}` | {_markdown_escape(value)} |")
                lines.append("")
        else:
            # Single image mode
            single = data if isinstance(data, dict) else {}
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            for key, value in sorted(single.items()):
                if key == "computed_intrinsics" and isinstance(value, dict):
                    for ckey, cval in sorted(value.items()):
                        lines.append(f"| `{ckey}` | {_markdown_escape(cval)} |")
                else:
                    lines.append(f"| `{key}` | {_markdown_escape(value)} |")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Saved Markdown: %s", path)
        return True
    except Exception as exc:
        logger.error("Failed to write Markdown: %s", exc)
        return False


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

    # For EXIF extraction, there are no complex numeric settings.
    # The main interactive bonus is the Markdown export option.
    print("EXIF extraction will output JSON by default.")
    print("You can optionally also save a human-readable Markdown copy.\n")

    if _ask_user("Also save a Markdown copy of the EXIF data?", default=False):
        args._save_markdown = True  # type: ignore[attr-defined]
    else:
        args._save_markdown = False  # type: ignore[attr-defined]

    print("\n" + "=" * 60)
    print("  Configuration complete!")
    print("=" * 60 + "\n")
    return args


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments, optionally enter interactive mode, and extract EXIF."""
    parser = _build_parser()
    args = parser.parse_args()

    # Interactive configuration overrides CLI flags
    if args.interactive:
        args = interactive_configure(args)

    # Resolve and validate input
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    images = collect_images(input_path)
    if not images:
        logger.error("No valid images found to process.")
        return 1

    is_batch = len(images) > 1 or input_path.is_dir()
    output_path = resolve_output_path(input_path, args.output, is_batch)

    # Summary before processing
    logger.info("Input:  %s", input_path)
    logger.info("Images: %d", len(images))
    logger.info("Output: %s", output_path)

    # Process images
    results: list[dict[str, Any]] = []
    success_count = 0

    for idx, img_path in enumerate(images, 1):
        logger.info("Processing %d / %d: %s", idx, len(images), img_path.name)
        exif_data = extract_exif(img_path)
        if exif_data is not None:
            results.append(exif_data)
            success_count += 1

    if not results:
        logger.error("No EXIF data could be extracted.")
        return 1

    # Prepare output data
    if is_batch:
        output_data: dict[str, Any] = {
            "source": str(input_path),
            "image_count": len(images),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "images": results,
        }
    else:
        output_data = results[0]

    # Write JSON
    if not write_json(output_data, output_path):
        return 1

    # Optional Markdown in interactive mode
    if getattr(args, "_save_markdown", False):
        md_path = output_path.with_suffix(".md")
        write_markdown(output_data, md_path)

    logger.info(
        "Finished. Successfully extracted EXIF from %d / %d images.",
        success_count,
        len(images),
    )

    return 0 if success_count == len(images) else 1


if __name__ == "__main__":
    sys.exit(main())
