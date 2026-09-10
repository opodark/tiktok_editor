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


def grip_part(cfg: LLMConfig, image: Path | str, hint: str = "") -> dict:
    """Quale parte del corpo tiene la presa sul palo e porta il peso.
    `hint` = testo di appoggio da pose.py (es. "polso sx vicino al palo").
    Ritorna {'parts': [...], 'load_bearing': str, 'on_pole': bool}."""
    q = (
        "Foto di pole dance. Guarda SOLO i punti dove il corpo TOCCA il palo di "
        "metallo verticale. Elenca al massimo 2 parti, quelle che davvero "
        "reggono il peso; se una parte non stringe il palo NON elencarla. "
        f"Parti possibili: {', '.join(_PARTS)}. Se nessuna tocca il palo, parts = []. "
        + (f"Indizio: {hint}. " if hint else "")
        + "Rispondi SOLO JSON con: parts (lista, max 2), load_bearing (il nome di UNA "
          'sola parte fra quelle di parts, quella che regge piu\' peso), on_pole (true/false). '
          'Esempio: {"parts": ["mano destra"], "load_bearing": "mano destra", "on_pole": true}'
    )
    try:
        out = _extract_json(ask_image(cfg, image, q, temperature=0.0))
    except Exception:  # noqa: BLE001
        return {"parts": [], "load_bearing": "", "on_pole": False}
    parts = [p for p in out.get("parts", []) if isinstance(p, str)][:2]
    return {"parts": parts,
            "load_bearing": str(out.get("load_bearing", "") or (parts[0] if parts else "")),
            "on_pole": bool(out.get("on_pole", bool(parts)))}


def name_pose(cfg: LLMConfig, image: Path | str, moves: Sequence[str],
              examples: Optional[Sequence[tuple]] = None) -> dict:
    """Nome della mossa dalla lista dell'istruttrice. Ritorna
    {'move','confidence'}. `move` = "sconosciuta" se non e' pole o se
    incerto. `held` (tenuta vs transizione) NON lo decide qui: viene dal
    movimento dei keypoint in pose.py. `examples` = coppie (percorso_png,
    nome_mossa) di riferimento dell'istruttrice (few-shot)."""
    lst = "; ".join(moves) if moves else "(nessuna lista fornita)"
    q = (
        "Questa foto e' di pole dance? Se NO, o se non sei sicuro della mossa, "
        'rispondi {"move": "sconosciuta", "confidence": 0}. '
        f"Altrimenti scegli il nome ESATTO da questa lista (nessun altro): {lst}. "
        'Rispondi SOLO JSON: {"move": "...", "confidence": 0.0-1.0}'
    )
    try:
        out = _extract_json(_ask_with_examples(cfg, image, q, examples))
    except Exception:  # noqa: BLE001
        return {"move": "sconosciuta", "confidence": 0.0}
    move = str(out.get("move", "") or "sconosciuta").strip()
    if moves and move.lower() != "sconosciuta" and move not in moves:
        # accetta match parziale, altrimenti scarta
        move = next((m for m in moves if m.lower() in move.lower()
                     or move.lower() in m.lower()), "sconosciuta")
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"move": move, "confidence": 0.0 if move == "sconosciuta" else conf}


def _ask_with_examples(cfg: LLMConfig, image, question, examples) -> str:
    """Come ask_image ma con qualche immagine-esempio etichettata prima."""
    if not examples:
        return ask_image(cfg, image, question, temperature=0.0)
    from openai import OpenAI
    client = OpenAI(api_key=cfg.api_key or "not-needed",
                    base_url=cfg.base_url.strip() or None, timeout=cfg.timeout)
    content = [{"type": "text", "text": "Esempi di riferimento:"}]
    for path, label in list(examples)[:6]:
        content.append({"type": "image_url", "image_url": {"url": _data_uri(path)}})
        content.append({"type": "text", "text": f"= {label}"})
    content.append({"type": "text", "text": question})
    content.append({"type": "image_url", "image_url": {"url": _data_uri(image)}})
    r = client.chat.completions.create(model=_vision_model(cfg), temperature=0.0,
                                       messages=[{"role": "user", "content": content}])
    return (r.choices[0].message.content or "").strip()


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
