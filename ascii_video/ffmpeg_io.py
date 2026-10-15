"""Read and write streaming BGR frames with external FFmpeg processes.

We deliberately avoid collecting decoded or encoded frames in a list.
"""
from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path


def probe(source: Path) -> dict:
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream found in {source}")
    fr = video.get("avg_frame_rate") or video.get("r_frame_rate") or "25/1"
    fps = Fraction(fr)
    if fps <= 0 or fps > 240:
        fps = Fraction(video.get("r_frame_rate") or "25/1")
    if fps <= 0 or fps > 240:
        raise ValueError(f"Could not determine a sensible FPS for {source}")
    audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    return {"width": int(video["width"]), "height": int(video["height"]),
            "fps": fps, "has_audio": audio,
            "duration": float(info.get("format", {}).get("duration", 0) or 0)}


def decoder(source: Path, width: int, height: int, fps: Fraction,
            start: float, duration: float | None):
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2"]
    if start:
        args.extend(["-ss", str(start)])
    args.extend(["-i", str(source)])
    if duration is not None:
        args.extend(["-t", str(duration)])
    # Force CFR here. The rawvideo pipe has no timestamps; matching FPS is essential.
    args.extend(["-map", "0:v:0", "-an", "-vf", f"fps={fps.numerator}/{fps.denominator}",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1024 * 1024)


def encoder(source: Path, output: Path, width: int, height: int, fps: Fraction,
            start: float, duration: float | None, encoder_name: str,
            audio_mode: str, preset: str, crf: int, log_path: Path):
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
            "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0"]
    if audio_mode != "none":
        if start:
            args.extend(["-ss", str(start)])
        if duration is not None:
            args.extend(["-t", str(duration)])
        args.extend(["-i", str(source)])
    args.extend(["-map", "0:v:0"])
    if audio_mode != "none":
        args.extend(["-map", "1:a:0?"])
    args.extend(["-c:v", encoder_name])
    if encoder_name == "libx264":
        args.extend(["-preset", preset, "-crf", str(crf), "-threads", "4"])
    elif encoder_name == "h264_nvenc":
        args.extend(["-preset", "p4", "-cq", str(crf), "-b:v", "0"])
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")
    args.extend(["-pix_fmt", "yuv420p"])
    if audio_mode == "copy":
        args.extend(["-c:a", "copy"])
    elif audio_mode == "aac":
        args.extend(["-c:a", "aac", "-b:a", "192k"])
    # A video pipe naturally ends at EOF. Do not use -shortest: AAC is sometimes
    # a few milliseconds shorter than video and would truncate the final frame.
    args.extend(["-movflags", "+faststart", str(output)])
    errfile = open(log_path, "wb")
    try:
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=errfile, bufsize=1024 * 1024)
    except Exception:
        errfile.close()
        raise
    return p, errfile


def exact_frame(stream, nbytes: int):
    """Return one frame (bytes), None at clean EOF; error at partial EOF."""
    data = bytearray(nbytes)
    view = memoryview(data)
    pos = 0
    while pos < nbytes:
        n = stream.readinto(view[pos:])
        if not n:
            if pos == 0:
                return None
            raise IOError(f"Truncated raw video frame: {pos}/{nbytes} bytes")
        pos += n
    return data
