"""Fixture condivise + skip automatico dei test che richiedono ffmpeg."""
from __future__ import annotations

import math
import shutil
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe non nel PATH")


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    """3 foto orizzontali sintetiche (1600x900) con un 'soggetto' in alto."""
    from PIL import Image, ImageDraw

    d = tmp_path / "media"
    d.mkdir()
    palette = ["#2a6df4", "#f4622a", "#2af49a"]
    for i, col in enumerate(palette):
        im = Image.new("RGB", (1600, 900), col)
        dr = ImageDraw.Draw(im)
        dr.ellipse((720, 90, 880, 250), fill="white", outline="black", width=6)
        dr.text((740, 300), f"FOTO {i}", fill="black")
        im.save(d / f"foto_{i}.jpg", quality=88)
    return d


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """WAV di 4s, ~120 BPM (tono + click sui beat)."""
    p = tmp_path / "song.wav"
    sr, dur = 22050, 4.0
    w = wave.open(str(p), "w")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    for n in range(int(sr * dur)):
        t = n / sr
        beat = 0.7 if (t % 0.5) < 0.02 else 0.0
        s = 0.3 * math.sin(2 * math.pi * 220 * t) + beat * math.sin(2 * math.pi * 880 * t)
        w.writeframes(struct.pack("<h", int(max(-1, min(1, s)) * 28000)))
    w.close()
    return p
