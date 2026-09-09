"""Caricamento e ispezione dei file media (foto e video) di input."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("autoedit")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass
class MediaItem:
    path: Path
    kind: str  # "image" | "video"
    duration: Optional[float] = None  # solo per i video, in secondi
    width: int = 0
    height: int = 0
    best_start: Optional[float] = None  # istante "interessante" della clip (solo video)

    @property
    def aspect(self) -> float:
        if self.height == 0:
            return 1.0
        return self.width / self.height


def _natural_key(path: Path):
    """Ordina 'img2' prima di 'img10' invece che alfabeticamente."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _best_start(path: Path, duration: float, n: int = 8) -> float:
    """Stima l'istante piu' "interessante" da cui far partire la clip:
    campiona n fotogrammi a bassa risoluzione nel 90% centrale e sceglie
    quello con piu' dettaglio + piu' movimento rispetto al precedente.
    In caso di problemi ritorna un valore ragionevole (15% della durata)."""
    if not duration or duration < 1.5:
        return 0.0
    lo, hi = duration * 0.06, duration * 0.94
    span = max(0.5, hi - lo)
    fallback = duration * 0.15
    tmp = Path(tempfile.mkdtemp(prefix="autoedit_scan_"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{lo:.2f}", "-t", f"{span:.2f}", "-i", str(path),
             "-vf", f"fps={n / span:.4f},scale=96:-1", "-frames:v", str(n),
             str(tmp / "f%02d.png"), "-loglevel", "error"],
            check=False, capture_output=True,
        )
        frames = sorted(tmp.glob("f*.png"))
        if len(frames) < 3:
            return fallback
        import numpy as np
        from PIL import Image
        arrs = [np.asarray(Image.open(f).convert("L"), dtype=float) for f in frames]
        detail = np.array([a.std() for a in arrs])
        motion = np.array([0.0] + [float(np.abs(arrs[i] - arrs[i - 1]).mean())
                                    for i in range(1, len(arrs))])
        norm = lambda v: (v - v.min()) / (float(np.ptp(v)) or 1.0)
        score = 0.65 * norm(motion) + 0.35 * norm(detail)
        best = int(np.argmax(score))
        t = lo + best / (len(arrs) - 1) * span
        return max(0.0, t - span / len(arrs) * 0.5)
    except Exception:  # noqa: BLE001
        return fallback
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def probe(path: Path) -> MediaItem:
    ext = path.suffix.lower()
    info = _ffprobe_json(path)
    vstream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    width = int(vstream["width"]) if vstream else 0
    height = int(vstream["height"]) if vstream else 0

    if ext in VIDEO_EXT:
        duration = float(info["format"].get("duration", 0.0))
        try:
            bs = _best_start(path, duration)
        except Exception:  # noqa: BLE001
            bs = None
        return MediaItem(path=path, kind="video", duration=duration, width=width, height=height,
                         best_start=bs)
    elif ext in IMAGE_EXT:
        return MediaItem(path=path, kind="image", duration=None, width=width, height=height)
    else:
        raise ValueError(f"Estensione non supportata: {path}")


def load_media(folder: Path, order_file: Optional[Path] = None) -> List[MediaItem]:
    """Carica tutti i media di una cartella.

    Se `order_file` e' passato (un file di testo con un nome file per
    riga) l'ordine dei clip nel montaggio segue quel file; altrimenti
    si usa l'ordine "naturale" dei nomi file (img1, img2, ..., img10).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    candidates = [
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT and p.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"Nessuna foto/video trovata in {folder}")

    if order_file:
        names = [line.strip() for line in Path(order_file).read_text().splitlines() if line.strip()]
        by_name = {p.name: p for p in candidates}
        ordered_paths = [by_name[n] for n in names if n in by_name]
    else:
        ordered_paths = sorted(candidates, key=_natural_key)

    return [probe(p) for p in ordered_paths]
