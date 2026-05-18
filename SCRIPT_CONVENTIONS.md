# Python CLI Script Specification

**Updated:** 2026-05-18

---

## 1. Purpose

This spec ensures every CLI script in this project follows a consistent structure, behavior, and user experience. The goal is modularity: any script should feel familiar to the user and be easy to maintain or extend.

---

## 2. File Naming

- **Format:** `snake_case.py` (all lowercase, underscores between words)
- **Pattern:** `[verb]_[noun].py` or `[function]_[detail].py`
- **Examples:**
  - `depth_map.py`
  - `invert_color.py`
  - `edge_detection.py`
  - `grid_stitcher.py`

---

## 3. Environment & Setup

### 3.1 Virtual Environment Handling
- **Detection:** Check if running inside a venv via:
  - `hasattr(sys, "real_prefix")`
  - `getattr(sys, "base_prefix", None) != sys.prefix`
  - `os.environ.get("VIRTUAL_ENV") is not None`
- **Existing `.venv`:** If `Path(__file__).parent / ".venv"` exists, automatically restart the script into it using `os.execv()` -- **no prompt**.
- **Missing venv:** Prompt user to create one (default: yes). If declined, exit with info message.
- **Venv path:** Always `Path(__file__).resolve().parent / ".venv"`

### 3.2 Dependency Management
- **Required packages map:** Dict mapping `import_name` -> `pip_package_name`
  ```python
  REQUIRED_PACKAGES = {
      "cv2": "opencv-python",
      "numpy": "numpy",
      "PIL": "Pillow",
  }
  ```
- **Check:** Try-import each module; collect missing packages.
- **Prompt:** List missing packages and ask to install (default: yes).
- **Install:** Run `[sys.executable, "-m", "pip", "install"] + packages`
- **Help bypass:** `--help` / `-h` must work **before** any environment checks or heavy imports.

### 3.3 Import Ordering
```python
# 1. Stdlib imports (argparse, logging, os, sys, pathlib)
# 2. Light-weight setup (no heavy imports)
# 3. CLI parser definition
# 4. Venv & dependency management helpers
# 5. If --help in sys.argv: print help and exit immediately
# 6. ensure_environment()
# 7. Heavy imports (cv2, numpy, PIL, torch, etc.)
# 8. Core logic
# 9. Interactive mode
# 10. Main entry point
```

---

## 4. CLI Structure

### 4.1 Argument Parser
- Use `argparse.ArgumentParser` with `RawDescriptionHelpFormatter`.
- Include `prog`, `description`, and `epilog` with usage examples.
- Use `%(prog)s` in examples instead of hardcoded names.

### 4.2 Standard Arguments (All Scripts)
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input` | Yes | -- | Input image file or folder |
| `output` | No | -- | Output file or folder (optional) |
| `-i`, `--interactive` | No | False | Step-by-step configuration mode |
| `--format` | No | `png` | Output format: `png`, `jpg`, `jpeg`, `bmp`, `tiff`, `webp` |

### 4.3 Script-Specific Arguments
- Add after standard arguments.
- Use `metavar` for value hints (e.g., `metavar="0-100"`).
- Provide sensible defaults and clear help text.
- Use `choices` for enums (e.g., `--method {feature,correlation}`).

### 4.4 Interactive Mode (`-i`)
- Triggered by `-i` / `--interactive` flag.
- Ask **key settings** first (the most important 1-3 options).
- Then ask: *"Do you want to specify additional settings?"* (default: no).
- Use `_ask_user()` helper for yes/no prompts.
- Validate numeric inputs with retry loops.
- Print a **"Configuration complete!"** banner before returning.

---

## 5. Input Handling

### 5.1 Auto-Detection (File vs Folder)
- If `input_path.is_file()` -> process single image.
- If `input_path.is_dir()` -> scan directory for supported images.
- If path doesn't exist -> log error and return exit code `1`.

### 5.2 Supported Image Formats
- Standard set: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`
- **Case-insensitive:** Match both `*.png` and `*.PNG`.
- Use `Path.glob()` for flat scan; `Path.rglob()` only if `--recursive` is supported.

### 5.3 Image Collection
- Return a **sorted** list of `Path` objects.
- Deduplicate results.
- Log count and warn if none found.

---

## 6. Output Rules

### 6.1 Default Naming
| Input Type | No Output | Output Specified |
|------------|-----------|------------------|
| Single file | `<stem>-<suffix>.<ext>` in same folder | Exact path given |
| Folder | Create `<input>/<subfolder>/` | Exact path given |

- `<suffix>` depends on script purpose:
  - Depth map -> `-depthmap`
  - Edge detection -> `-edges`
  - Color invert -> `-inverted`
  - Grid stitch -> `stitched.png` (in folder)
- Extension defaults to `--format` value.

### 6.2 Path Resolution Logic
- If user path ends with `/` or `\\` -> treat as directory.
- If user passes a file path but multiple images are being processed -> **error**.
- Create parent directories automatically (`mkdir(parents=True, exist_ok=True)`).

---

## 7. Core Processing Patterns

### 7.1 Image Loading
- Use `cv2.imread(str(path))` for OpenCV pipelines.
- Use `PILImage.open(path)` for format conversion or saving.
- Load in BGR (OpenCV default); convert to RGB/gray as needed.

### 7.2 Batch Processing
- Generate `(input_path, output_path)` pairs before processing.
- Iterate pairs; process each independently.
- Log progress: *"Processed X / Y images"*
- Track success count vs total.
- Return exit code `0` if all succeeded, `1` if any failed.

### 7.3 Per-Image Error Handling
- Wrap processing in try/except.
- Log error with image path; continue to next image.
- Don't crash the whole batch on one failure.

---

## 8. Logging Standards

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("script_name")
```

### 8.1 Required Log Events
- Input path and image count
- Output path
- Key settings (model, thresholds, method, etc.)
- Progress updates (for multi-image operations)
- Completion summary: *"Successfully processed X / Y images"*

---

## 9. Code Quality

### 9.1 File Structure
```python
#!/usr/bin/env python3
"""
Created by OpenCode.
Initialized: YYYY-MM-DD
Updated: YYYY-MM-DD

Short description.

Long description with usage examples.

Environment notes.
"""

from __future__ import annotations

# 1. Stdlib imports
# 2. Light-weight setup (no heavy imports)
# 3. CLI parser definition
# 4. Venv & dependency management
# 5. Heavy imports (after environment is confirmed)
# 6. Core logic
# 7. Interactive mode
# 8. Main entry point
```

### 9.2 Section Dividers
```python
# ---------------------------------------------------------------------------
# Section Name
# ---------------------------------------------------------------------------
```

### 9.3 Type Hints
- Use `from __future__ import annotations`.
- Annotate all function signatures.
- Use `|` syntax for unions (e.g., `str | None`).
- Use `tuple[...]`, `list[...]`, `dict[...]`.

### 9.4 Docstrings
- Triple double-quotes.
- Include **Args**, **Returns**, and **Raises** for non-trivial functions.
- One-liner acceptable for trivial helpers (`_ask_user`, `_in_virtual_env`).

---

## 10. Consistency Checklist

Before any script is considered complete, verify:

- [ ] `--help` works immediately without triggering venv or dependency checks
- [ ] Existing `.venv` is auto-detected and restarted into without prompting
- [ ] Missing dependencies prompt user before installing
- [ ] `input` accepts both files and folders with auto-detection
- [ ] Output follows default naming rules when not specified
- [ ] Interactive mode (`-i`) asks key settings, then offers advanced settings
- [ ] Logging format matches the standard (`HH:MM:SS | LEVEL    | message`)
- [ ] All image formats are case-insensitive
- [ ] Script returns `0` on full success, `1` on partial or full failure
- [ ] Type hints and docstrings are present on all functions
- [ ] File uses `snake_case.py` naming

