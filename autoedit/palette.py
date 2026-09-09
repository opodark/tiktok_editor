"""Colori dominanti di una o piu' immagini -> lista di #esadecimali.

Serve a dare CONTESTO visivo all'LLM (grafica e color grade coerenti coi
media) senza bisogno di un modello multimodale: si campiona e si
raggruppano i pixel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

_NAMES = [
    ((0, 0, 0), "nero"), ((255, 255, 255), "bianco"), ((128, 128, 128), "grigio"),
    ((200, 30, 30), "rosso"), ((240, 140, 30), "arancio"), ((240, 220, 40), "giallo"),
    ((60, 180, 75), "verde"), ((40, 120, 240), "blu"), ((110, 40, 200), "viola"),
    ((240, 90, 180), "rosa"), ((120, 72, 40), "marrone"), ((30, 200, 200), "ciano"),
]


def _name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return min(_NAMES, key=lambda c: (c[0][0] - r) ** 2 + (c[0][1] - g) ** 2 + (c[0][2] - b) ** 2)[1]


def _hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(max(0, min(255, v))) for v in rgb))


def dominant_colors(paths: Iterable[Path | str], k: int = 5, sample: int = 120) -> list[str]:
    """Ritorna fino a `k` colori #esadecimali, dal piu' frequente, saltando
    quelli quasi identici fra loro."""
    pts: list[np.ndarray] = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        im.thumbnail((sample, sample))
        pts.append(np.asarray(im, dtype=np.float32).reshape(-1, 3))
    if not pts:
        return []
    data = np.concatenate(pts, axis=0)

    # k-means "povero" ma sufficiente: pochi punti, poche iterazioni
    rng = np.random.default_rng(0)
    cent = data[rng.choice(len(data), size=min(k * 2, len(data)), replace=False)]
    for _ in range(8):
        d = ((data[:, None, :] - cent[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        new = np.array([data[lab == i].mean(0) if np.any(lab == i) else cent[i]
                        for i in range(len(cent))])
        if np.allclose(new, cent, atol=1.0):
            cent = new
            break
        cent = new

    counts = np.bincount(lab, minlength=len(cent))
    order = np.argsort(counts)[::-1]
    out: list[str] = []
    picked: list[np.ndarray] = []
    for i in order:
        if counts[i] == 0:
            continue
        c = cent[i]
        if any(np.sqrt(((c - q) ** 2).sum()) < 38 for q in picked):
            continue
        picked.append(c)
        out.append(_hex(c))
        if len(out) >= k:
            break
    return out


def describe(paths: Iterable[Path | str], k: int = 4) -> str:
    """Riga pronta per il prompt: '#6E1FD6 viola, #43FF13 verde, ...'."""
    cols = dominant_colors(paths, k=k)
    return ", ".join(f"{h} {_name(tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)))}" for h in cols)
