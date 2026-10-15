# ASCII Video Lab

A **streaming** Python / OpenCV / FFmpeg pipeline for rendering videos into ASCII. Optimized for setups with a GPU

Unlike other video to ASCII converters, this implementation only streams one decoded frame at a time, allowing you to process longer videos without running into issues with RAM

This repository also contains a variety of methods to deal with noisy input video (grainy, scanlines, glitch etc) to give fine grained control over the render



## Installation

### Setup

``` bash
# Install packages through your package manager (NOTE: this is the usage for Arch, you will need to substitute for your own package manger (e.g. apt))
sudo pacman -S ffmpeg ttf-dejavu 
python -m pip install --upgrade pip
# PyTorch CUDA 13.0 wheel; works with an NVIDIA driver that supports the CUDA runtime:
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
# Check your torch/CUDA install:
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
```

Check that `torch.cuda.is_available()` prints `True` before running. Otherwise PyTorch may have been installed with a CPU-only build. See [PyTorch official installation instructions](https://pytorch.org/get-started/locally/) if you need a different CUDA wheel. A CUDA toolkit compiler (`nvcc`) is **not** required to use the pre-built wheel.


## Usage:

### Run a 10-second preview first

```bash
python -m ascii_video ~/Videos/test.mp4 ~/Videos/ascii-preview.mp4 \
    --start 30 --duration 10 --debug-frame 50
```

This writes `ascii-preview.mp4` and saves `ascii-debug/` images (`original.png`, `denoised.png`, `edges.png`, `ascii.png`, `composite.png`).

### Whole video

```bash
python -m ascii_video ~/Videos/test.mp4 ~/Videos/ascii-pip.mp4
```

Original audio is copied without re-encoding by default (`--audio copy`). Set `--audio aac` if the source audio codec is incompatible with MP4.

### Useful adjustments

```bash
# Stronger clean-up, fewer noisy edges:
python -m ascii_video ~/Videos/test.mp4 ~/Videos/clean.mp4 \
  --start 30 --duration 10 --spatial bilateral \
  --temporal-alpha 0.15 --motion-threshold 20 \
  --edge-threshold 130 --stability 10 --contrast 1.2

# Comparison: traditional ASCII only
python -m ascii_video ~/Videos/test.mp4 ~/Videos/tonal.mp4 \
  --start 30 --duration 10 --mode luminance

# Just the ASCII image (no inset), coloured phosphor
python -m ascii_video ~/Videos/test.mp4 ~/Videos/amber.mp4 \
  --start 30 --duration 10 --inset-width 0 --palette amber

# NVIDIA GPU encoding (requires working NVENC in your FFmpeg driver stack)
python -m ascii_video ~/Videos/test.mp4 ~/Videos/fast.mp4 \
  --encoder h264_nvenc
```

Parameter notes:

| Option | Default | Effect |
|---|---:|---|
| `--mode` | `hybrid` | `hybrid`, `luminance`, or `edges` |
| `--edge-threshold` | `80` | Higher = fewer noisy edge glyphs |
| `--temporal-alpha` | `0.25` | Lower = more temporal noise suppression; may ghost |
| `--motion-threshold` | `12` | Higher = suppress more changes; may smear moving subjects |
| `--stability` | `5` | Hold chars when brightness changes are below this amount |
| `--cell-width`, `--cell-height` | `10`, `16` | Text density; width and height must be divisible |
| `--analysis-scale` | `4` | Spatial resolution used for filtering relative to character grid |
| `--inset-width` | `420` | Original video width in pixels; `0` disables inset |
| `--background` | `cover` | `cover` crops, `contain` letterboxes source for ASCII |
| `--width`, `--height` | `1280`, `720` | Output resolution |
| `--start`, `--duration` | `0`, full | Trim a preview segment |

## Technical notes

- Input is decoded via the **system FFmpeg**. This avoids OpenCV codec-plugin compatibility issues experienced with the old Rust release.
- By default FFmpeg's `fps` video filter converts video to a constant rate, based on the source's reported average FPS; this avoids accumulating timestamp errors but may duplicate/drop frames for variable-frame-rate input. No audio resampling is done when using `--audio copy`.
- The original inset and ASCII background come from the **same decoded frame**, so they cannot drift independently.
- The spatial cleaner operates at a small, ASCII-appropriate resolution (512×180 by default), which avoids expensive work at 4K.
- The current temporal filter is motion-gated EMA, not optical-flow-compensated denoising. Moving noisy objects may still flicker; flow-based alignment is a good v0.2 feature.
- The glyph picker uses a luminance ramp and strong Sobel edge orientations, followed by character-grid hysteresis.
- If input audio exceeds the video by a little, FFmpeg may produce an audio tail. The next release can add stricter audio trimming when needed.
- CPU vs NVENC performance varies with hardware; benchmark before assuming GPU will always be faster.

## Troubleshooting

- `File not found: font`: supply `--font /usr/share/fonts/TTF/DejaVuSansMono.ttf` or another monospace TTF/OTF.
- FFmpeg fails to copy an unusual audio codec: rerun with `--audio aac`.
- `ERROR: output exists`: choose a new filename or use `--overwrite`.
- Broken process leaves `*.incomplete.mp4` and possibly `*.ffmpeg-error.log`: inspect the log; use `--overwrite` to retry.
