"""Modello di visione LOCALE (via Ollama / endpoint OpenAI-compatibile).

Serve a "leggere" un fotogramma e dire cose utili al montaggio tutorial:
- che mossa di pole/exotic e' (da una lista fornita dall'istruttrice)
- quale parte del corpo regge la presa in quel momento
- una frase-consiglio breve per l'allieva

Gira SOLO su pochi fotogrammi candidati (i "fermi" trovati da pose.py),
non su tutto il video: un 7B multimodale ci mette 1-3 s a frame.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional, Sequence

from .brief import LLMConfig, _extract_json

DEFAULT_VISION_MODEL = "qwen2.5vl:7b"


def _data_uri(image: Path | str) -> str:
    p = Path(image)
    ext = p.suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


def _vision_model(cfg: LLMConfig) -> str:
    return (getattr(cfg, "vision_model", "") or "").strip() or DEFAULT_VISION_MODEL


def ask_image(cfg: LLMConfig, image: Path | str, question: str,
              system: Optional[str] = None, temperature: float = 0.2) -> str:
    """Una domanda su un'immagine -> testo della risposta. Endpoint
    OpenAI-compatibile (Ollama: /v1). Usa `cfg.vision_model` o il default."""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Manca 'openai'. Installa:  pip install openai") from e

    client = OpenAI(api_key=cfg.api_key or "not-needed",
                    base_url=cfg.base_url.strip() or None, timeout=cfg.timeout)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": _data_uri(image)}},
    ]})
    r = client.chat.completions.create(model=_vision_model(cfg), temperature=temperature,
                                       messages=messages)
    return (r.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------
# Domande specifiche per il montaggio tutorial
# --------------------------------------------------------------------------
_PARTS = ["mano sinistra", "mano destra", "piede sinistro", "piede destro",
          "incavo del ginocchio", "coscia", "ascella", "caviglia", "schiena"]


def grip_part(cfg: LLMConfig, image: Path | str) -> dict:
    """Quale/i parte/i del corpo tiene la presa sul palo, e quale porta il
    peso. Ritorna {'parts': [...], 'load_bearing': str, 'on_pole': bool}."""
    q = (
        "Nella foto una persona su un palo da pole dance. "
        "Quali parti del corpo sono a CONTATTO col palo e reggono la posizione? "
        f"Scegli da: {', '.join(_PARTS)}. "
        'Rispondi SOLO JSON: {"parts": ["..."], "load_bearing": "...", "on_pole": true/false}'
    )
    try:
        return _extract_json(ask_image(cfg, image, q))
    except Exception:  # noqa: BLE001
        return {"parts": [], "load_bearing": "", "on_pole": False}


def name_pose(cfg: LLMConfig, image: Path | str, moves: Sequence[str]) -> dict:
    """Nome della mossa dalla lista dell'istruttrice + se e' una posa
    'tenuta' (pulita) o una transizione. {'move','held','confidence'}."""
    lst = "; ".join(moves) if moves else "(nessuna lista fornita)"
    q = (
        "Foto di pole dance / exotic. Che mossa e'? "
        f"Scegli il nome PIU' probabile da questa lista: {lst}. "
        "Dimmi anche se e' una posa tenuta ferma e pulita (held=true) o un "
        'passaggio (held=false). Rispondi SOLO JSON: '
        '{"move": "...", "held": true/false, "confidence": 0.0-1.0}'
    )
    try:
        out = _extract_json(ask_image(cfg, image, q))
    except Exception:  # noqa: BLE001
        return {"move": "", "held": False, "confidence": 0.0}
    out.setdefault("move", "")
    out.setdefault("held", False)
    try:
        out["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0.0))))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    return out


def cue(cfg: LLMConfig, image: Path | str, move: str = "") -> str:
    """Una frase-consiglio breve (max ~12 parole) per l'allieva."""
    m = f' per la mossa "{move}"' if move else ""
    q = (f"Sei un'istruttrice di pole dance. Guarda la foto e da' UN consiglio "
         f"tecnico brevissimo{m} (max 12 parole, in italiano, imperativo). "
         "Solo la frase, niente altro.")
    txt = ask_image(cfg, image, q, temperature=0.4)
    return re.sub(r"\s+", " ", txt).strip().strip('"')[:120]


def available(cfg: LLMConfig) -> bool:
    """True se il modello di visione risponde (per capabilities())."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.api_key or "not-needed",
                        base_url=cfg.base_url.strip() or None, timeout=5.0)
        names = {m.id for m in client.models.list().data}
        return _vision_model(cfg) in names or any(
            _vision_model(cfg).split(":")[0] in n for n in names)
    except Exception:  # noqa: BLE001
        return False
