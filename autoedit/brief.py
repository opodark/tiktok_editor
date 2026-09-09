"""Brief in linguaggio naturale -> impostazioni di montaggio, via LLM.

Supporta due provider, scelti a runtime:
- "openai"    : qualunque endpoint compatibile con l'API OpenAI, anche
                LOCALE (Ollama, LM Studio, llama.cpp server, vLLM, ...).
                Usa il pacchetto ufficiale `openai` con `base_url` custom.
- "anthropic" : Claude via il pacchetto ufficiale `anthropic`.

Entrambi i pacchetti sono importati in modo pigro: se manca quello del
provider scelto, l'errore lo dice chiaramente. Niente shim fatti a mano.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .effects import COLOR_STYLES, EDIT_STYLES, IMPACT_EFFECTS, MOTION_MODES

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "llm_config.json"

ASPECTS = ["9:16", "4:5", "1:1", "16:9"]
TRANSITION_MODES = ["auto", "cut", "fade", "slide", "zoom", "chaos"]
TITLE_POSITIONS = ["top", "center", "bottom"]


@dataclass
class LLMConfig:
    provider: str = "openai"      # "openai" (compatibile) | "anthropic"
    # default: Ollama in locale. Vuoto = endpoint di default del provider.
    base_url: str = "http://localhost:11434/v1"
    model: str = ""              # es. "qwen2.5:7b", "hf.co/utente/repo:Q4_K_M", "claude-sonnet-5"
    asset_model: str = ""        # modello per la grafica generata (vuoto = usa `model`);
                                 # un modello "coder" fa SVG piu' puliti
    api_key: str = ""
    timeout: float = 180.0        # i modelli locali possono essere lenti


def load_llm_config() -> LLMConfig:
    if _CONFIG_PATH.is_file():
        try:
            raw = json.loads(_CONFIG_PATH.read_text())
            return LLMConfig(**{**asdict(LLMConfig()), **{k: raw[k] for k in raw
                                                          if k in asdict(LLMConfig())}})
        except Exception:  # noqa: BLE001
            pass
    return LLMConfig()


def save_llm_config(cfg: LLMConfig) -> None:
    _CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
def _schema_text() -> str:
    return (
        "Rispondi SOLO con un oggetto JSON valido, senza testo attorno. "
        "Tutti i campi sono opzionali: includi solo quelli che vuoi cambiare.\n"
        f'  "edit_style": uno di {sorted(EDIT_STYLES)} '
        "(ricetta di montaggio: sceglila quasi sempre invece dei singoli parametri; "
        "clean=stacchi netti puliti, hype=veloce con effetti, smooth=morbido con transizioni, "
        "glitch=disturbo digitale, retro=pellicola)\n"
        f'  "style": uno di {sorted(COLOR_STYLES)}\n'
        f'  "motion": uno di {MOTION_MODES}\n'
        '  "motion_intensity": numero 0.3-2.2\n'
        f'  "transition_mode": uno di {TRANSITION_MODES}\n'
        '  "variety": numero 0-1 (alto = ogni clip diversa, anti-monotonia)\n'
        f'  "impact_effects": lista, sottoinsieme di {sorted(IMPACT_EFFECTS)}\n'
        '  "grain": numero 0-1\n'
        '  "vignette": true/false\n'
        '  "chromatic": true/false\n'
        '  "slowmo": true/false (slow-motion e accelerazioni sulle clip video)\n'
        '  "hold_prob": numero 0-0.4 (quanto spesso una clip viene tenuta 2 beat)\n'
        '  "cuts_per_beat": 0.5, 1 oppure 2\n'
        f'  "aspect": uno di {ASPECTS}\n'
        '  "strong_beats_only": true/false\n'
        '  "dynamic_pacing": true/false\n'
        '  "xfade_ms": intero 0-400 (sotto 70 = stacchi netti)\n'
        '  "hook_hold": numero 0-2 (secondi di fermo iniziale)\n'
        '  "title_text": stringa (o "")\n'
        f'  "title_pos": uno di {TITLE_POSITIONS}\n'
    )


def _system_prompt() -> str:
    return (
        "Sei un montatore esperto di reel verticali (TikTok / Reels / Shorts). "
        "Dato un brief e il contesto (brano musicale + media disponibili), scegli "
        "le impostazioni di montaggio piu' adatte, evitando la monotonia: se il "
        "brief chiede energia, alza variety e dynamic_pacing e attiva effetti "
        "d'impatto; se chiede calma, usa transizioni morbide, movimento lieve e "
        "poca varieta'. Adatta il formato e lo stile colore al tono richiesto.\n\n"
        + _schema_text()
    )


def _user_prompt(brief: str, context: dict) -> str:
    names = ", ".join(context.get("names", [])[:40]) or "(non ancora analizzati)"
    return (
        f"BRIEF DELL'UTENTE:\n{brief.strip()}\n\n"
        "CONTESTO:\n"
        f"- brano: {context.get('bpm', '?')} BPM, durata {context.get('duration', '?')}s, "
        f"{context.get('downbeats', '?')} downbeat\n"
        f"- media: {context.get('n_media', '?')} file "
        f"({context.get('n_images', '?')} foto, {context.get('n_videos', '?')} video)\n"
        f"- nomi file: {names}"
    )


# --------------------------------------------------------------------------
# Parsing / validazione
# --------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?|```", "", text or "").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("La risposta del modello non contiene JSON.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("La risposta del modello contiene un JSON incompleto.")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _num(v):
    return float(str(v).replace(",", "."))


def validate_overrides(raw: dict) -> dict:
    """Tiene solo i campi noti, con valori nei range consentiti."""
    out: dict = {}
    if raw.get("edit_style") in EDIT_STYLES:
        out["edit_style"] = raw["edit_style"]
    if raw.get("style") in COLOR_STYLES:
        out["style"] = raw["style"]
    if raw.get("motion") in MOTION_MODES:
        out["motion"] = raw["motion"]
    if raw.get("transition_mode") in TRANSITION_MODES:
        out["transition_mode"] = raw["transition_mode"]
    if raw.get("aspect") in ASPECTS:
        out["aspect"] = raw["aspect"]
    if raw.get("title_pos") in TITLE_POSITIONS:
        out["title_pos"] = raw["title_pos"]
    if isinstance(raw.get("impact_effects"), list):
        out["impact_effects"] = [e for e in raw["impact_effects"] if e in IMPACT_EFFECTS]
    if isinstance(raw.get("title_text"), str):
        out["title_text"] = raw["title_text"][:120]
    for b in ("vignette", "chromatic", "strong_beats_only", "dynamic_pacing", "slowmo"):
        if isinstance(raw.get(b), bool):
            out[b] = raw[b]
    for key, lo, hi in (("motion_intensity", 0.3, 2.2), ("variety", 0.0, 1.0),
                         ("grain", 0.0, 1.0), ("hook_hold", 0.0, 2.0), ("hold_prob", 0.0, 0.4)):
        if key in raw:
            try:
                out[key] = _clamp(_num(raw[key]), lo, hi)
            except (TypeError, ValueError):
                pass
    if "xfade_ms" in raw:
        try:
            out["xfade_ms"] = int(_clamp(_num(raw["xfade_ms"]), 0, 400))
        except (TypeError, ValueError):
            pass
    if "cuts_per_beat" in raw:
        try:
            cpb = _num(raw["cuts_per_beat"])
            if cpb in (0.5, 1.0, 2.0):
                out["cuts_per_beat"] = cpb
        except (TypeError, ValueError):
            pass
    return out


# --------------------------------------------------------------------------
# Chiamate LLM (pacchetti ufficiali, import pigro)
# --------------------------------------------------------------------------
def _chat_openai(cfg: LLMConfig, system: str, user: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Manca il pacchetto 'openai'. Installa:  pip install openai") from e
    client = OpenAI(
        api_key=cfg.api_key or "not-needed",
        base_url=cfg.base_url.strip() or None,
        timeout=cfg.timeout,
    )
    # streaming: i modelli locali lenti non fanno scattare il read-timeout
    stream = client.chat.completions.create(
        model=cfg.model,
        temperature=0.5,
        stream=True,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    return "".join(parts)


def _chat_anthropic(cfg: LLMConfig, system: str, user: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Manca il pacchetto 'anthropic'. Installa:  pip install anthropic") from e
    kwargs: dict = {"timeout": cfg.timeout}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url.strip():
        kwargs["base_url"] = cfg.base_url.strip()
    client = anthropic.Anthropic(**kwargs)
    with client.messages.stream(
        model=cfg.model or "claude-opus-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def suggest_overrides(cfg: LLMConfig, brief: str, context: dict) -> dict:
    """Chiede all'LLM le impostazioni per il brief e ritorna solo gli
    override validi (chiavi interne di RenderConfig)."""
    if not brief or not brief.strip():
        raise ValueError("Scrivi un brief prima.")
    if not cfg.model.strip():
        raise ValueError("Imposta il nome del modello nella sezione connessione.")
    chat = _chat_anthropic if cfg.provider == "anthropic" else _chat_openai
    text = chat(cfg, _system_prompt(), _user_prompt(brief, context))
    return validate_overrides(_extract_json(text))


# --------------------------------------------------------------------------
# Grafica generata: brief -> AssetSpec
# --------------------------------------------------------------------------
def _asset_system_prompt() -> str:
    from .assets import asset_schema_text
    return (
        "Sei un art director di grafica per reel verticali (TikTok / Reels). "
        "Dato un brief, produci UN elemento grafico: un logo/wordmark, una card "
        "titolo, un lower-third, un @handle, un badge, un cartellino prezzo, una "
        "call-to-action, oppure -- se serve una forma libera -- dell'SVG. "
        "Stile pulito e ad alto contrasto, leggibile su qualsiasi sfondo. "
        "Preferisci SEMPRE un template ai disegni SVG. Colori in #esadecimale.\n\n"
        + asset_schema_text()
    )


def suggest_asset(cfg: LLMConfig, brief: str, context: dict | None = None):
    """Chiede all'LLM un elemento grafico e ritorna un AssetSpec validato.

    Usa `cfg.asset_model` se impostato (un modello 'coder' fa SVG migliori),
    altrimenti `cfg.model`.
    """
    from dataclasses import replace

    from .assets import validate_asset_spec

    if not brief or not brief.strip():
        raise ValueError("Scrivi cosa vuoi generare.")
    use = replace(cfg, model=cfg.asset_model.strip() or cfg.model)
    if not use.model.strip():
        raise ValueError("Imposta il modello (o il 'modello grafica') nelle Impostazioni.")

    ctx = context or {}
    hint = ""
    if ctx.get("aspect"):
        hint += f"\nFormato del video: {ctx['aspect']} (adatta width/height)."
    if ctx.get("palette"):
        hint += f"\nColori coerenti col video: {ctx['palette']}."
    user = f"BRIEF:\n{brief.strip()}{hint}"

    chat = _chat_anthropic if use.provider == "anthropic" else _chat_openai
    text = chat(use, _asset_system_prompt(), user)
    return validate_asset_spec(_extract_json(text))
