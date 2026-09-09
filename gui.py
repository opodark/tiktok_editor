#!/usr/bin/env python3
"""Interfaccia grafica per autoedit (gira nel browser, tutto in locale).

Avvio:
    pip install -r requirements-gui.txt
    python gui.py

Flusso in due passi:
  1. ANALIZZA  -> carichi foto/video + canzone, vedi miniature, forma
     d'onda con i beat e i dati del brano.
  2. GENERA    -> scegli effetti / ritmo / formato, eventualmente riordini
     o escludi media nella tabella, e monti il video.
"""
from __future__ import annotations

import json
import logging
import random
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import gradio as gr
except ImportError:
    sys.exit("Manca 'gradio'. Installalo con:  pip install -r requirements-gui.txt")

import numpy as np
from PIL import Image, ImageDraw

from autoedit.beats import detect_beats
from autoedit.brief import LLMConfig, load_llm_config, save_llm_config, suggest_overrides
from autoedit.effects import COLOR_LABELS, EDIT_STYLE_LABELS, IMPACT_LABELS, MOTION_LABELS
from autoedit.media import IMAGE_EXT, VIDEO_EXT
from autoedit.pipeline import ASPECTS, RenderConfig, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
for _noisy in ("numba", "librosa", "matplotlib"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# --- etichette "umane" <-> valori interni ---------------------------------
SPEEDS = {
    "Lento — 1 taglio ogni 2 beat": 0.5,
    "Normale — 1 taglio a beat": 1.0,
    "Veloce — 2 tagli a beat": 2.0,
}
ASPECT_LABELS = {
    "Verticale 9:16 (TikTok / Reels / Shorts)": "9:16",
    "Verticale 4:5 (feed Instagram)": "4:5",
    "Quadrato 1:1": "1:1",
    "Orizzontale 16:9": "16:9",
}
TRANSITIONS = {
    "Automatiche": "auto",
    "Solo stacchi netti": "cut",
    "Solo dissolvenze": "fade",
    "Solo slide": "slide",
    "Solo zoom": "zoom",
    "Caotiche": "chaos",
}
WM_POS = {
    "Basso a destra": "br", "Basso a sinistra": "bl",
    "Alto a destra": "tr", "Alto a sinistra": "tl",
}
TITLE_POS = {"In alto": "top", "Al centro": "center", "In basso": "bottom"}

STYLE_BY_LABEL = {v: k for k, v in COLOR_LABELS.items()}
MOTION_BY_LABEL = {v: k for k, v in MOTION_LABELS.items()}
IMPACT_BY_LABEL = {v: k for k, v in IMPACT_LABELS.items()}
# interno -> etichetta (per applicare i suggerimenti del Brief AI)
SPEED_BY_VALUE = {v: k for k, v in SPEEDS.items()}
ASPECT_BY_VALUE = {v: k for k, v in ASPECT_LABELS.items()}
TRANS_BY_VALUE = {v: k for k, v in TRANSITIONS.items()}
TITLE_BY_VALUE = {v: k for k, v in TITLE_POS.items()}
EDIT_STYLE_BY_LABEL = {v: k for k, v in EDIT_STYLE_LABELS.items()}
LLM_PROVIDERS = {"Compatibile OpenAI (anche locale)": "openai", "Anthropic (Claude)": "anthropic"}
PROVIDER_LABEL = {v: k for k, v in LLM_PROVIDERS.items()}

ALLOWED_EXT = sorted(IMAGE_EXT | VIDEO_EXT)
TABLE_HEADERS = ["#", "file", "includi", "ordine"]

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
# campi salvati in un preset (ordine = ordine degli output di "Carica")
PRESET_FIELDS = [
    "edit_style", "style", "speed", "aspect", "transition", "motion", "motion_intensity",
    "variety", "impact", "grain", "vignette", "chromatic", "slowmo", "hold_prob", "title_text",
    "title_pos", "title_start", "title_dur", "wm_pos", "wm_scale", "wm_opacity", "hook_hold",
    "audio_start", "beat_offset", "xfade_ms", "snap_onsets", "dynamic_pacing", "strong_only",
    "shuffle", "max_dur", "seed",
]


def _list_presets() -> list[str]:
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def _as_path(f) -> Path:
    return Path(getattr(f, "name", f))


def _make_thumb(path: Path, kind: str, out: Path) -> None:
    if kind == "video":
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "0", "-i", str(path), "-frames:v", "1",
             "-vf", "scale=240:-2", str(out), "-loglevel", "error"],
            check=False,
        )
    if not out.exists():
        try:
            im = Image.open(path)
            im.thumbnail((240, 426))
            im.convert("RGB").save(out, "JPEG", quality=85)
        except Exception:  # noqa: BLE001
            pass


def _waveform_png(audio_path: Path, beat_info, out: Path, w: int = 1200, h: int = 240) -> None:
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=8000, mono=True)
    except Exception:  # noqa: BLE001
        y, sr = np.zeros(8000), 8000
    dur = max(1e-3, len(y) / sr)
    env = np.abs(y)
    step = max(1, len(env) // w)
    cols = np.array([env[i * step:(i + 1) * step].max() if (i * step) < len(env) else 0.0
                     for i in range(w)])
    cols = cols / (cols.max() or 1.0)

    img = Image.new("RGB", (w, h), "#0f1117")
    d = ImageDraw.Draw(img)
    mid = h // 2
    for x, v in enumerate(cols):
        bar = int(v * (h * 0.44))
        d.line([(x, mid - bar), (x, mid + bar)], fill="#5b8def")
    for t in getattr(beat_info, "beat_times", []):
        d.line([(int(t / dur * w), 0), (int(t / dur * w), h)], fill="#343a46")
    for t in getattr(beat_info, "downbeat_times", []):
        d.line([(int(t / dur * w), 0), (int(t / dur * w), h)], fill="#e0b341")
    for t in getattr(beat_info, "strong_times", []):
        d.line([(int(t / dur * w), h - 16), (int(t / dur * w), h)], fill="#e0553b")
    img.save(out)


def _rows_from_table(table) -> list[list]:
    if table is None:
        return []
    if hasattr(table, "values"):
        return table.values.tolist()
    return [list(r) for r in table]


# --- logica pura (nessun oggetto Gradio) ---------------------------------
def _do_analyze(files, audio):
    if not files:
        raise gr.Error("Carica almeno una foto o un video.")
    if not audio:
        raise gr.Error("Carica un file audio (mp3 o wav).")

    work = Path(tempfile.mkdtemp(prefix="autoedit_gui_"))
    media_dir = work / "media"
    media_dir.mkdir()

    metas = []
    for i, f in enumerate(files):
        src = _as_path(f)
        if src.suffix.lower() not in IMAGE_EXT | VIDEO_EXT:
            continue
        dst = media_dir / f"{i:03d}_{src.name}"
        shutil.copy(src, dst)
        kind = "video" if src.suffix.lower() in VIDEO_EXT else "image"
        thumb = work / f"thumb_{i:03d}.jpg"
        _make_thumb(dst, kind, thumb)
        metas.append({"name": dst.name, "orig": src.name, "kind": kind,
                      "thumb": str(thumb) if thumb.exists() else str(dst)})
    if not metas:
        raise gr.Error("Nessun file valido. Formati: " + ", ".join(e.lstrip(".") for e in ALLOWED_EXT))

    audio_path = _as_path(audio)
    bi = detect_beats(audio_path)
    wave = work / "wave.png"
    _waveform_png(audio_path, bi, wave)

    gallery = [(m["thumb"], f'{i} · {m["orig"]}') for i, m in enumerate(metas)]
    table = [[i, m["orig"], True, i] for i, m in enumerate(metas)]
    info = (
        f"### Brano · {bi.tempo:.0f} BPM\n"
        f"- {len(bi.beat_times)} beat · {len(bi.downbeat_times)} downbeat (gialli) · "
        f"{len(bi.strong_times)} beat forti (rossi) · {len(bi.onset_times)} transienti\n"
        f"- durata {bi.duration:.1f}s · **{len(metas)} media** caricati"
    )
    state = {"work": str(work), "media_dir": str(media_dir),
             "audio": str(audio_path), "metas": metas,
             "beat": {"bpm": round(float(bi.tempo)), "duration": round(float(bi.duration), 1),
                      "downbeats": int(len(bi.downbeat_times))}}
    return state, gallery, table, str(wave), info


def _do_generate(state, p: dict, force_seed, progress) -> tuple[str, str]:
    if not state:
        raise gr.Error("Premi prima «Analizza».")
    media_dir = Path(state["media_dir"])
    audio_path = Path(state["audio"])
    metas = state["metas"]
    work = Path(state["work"])

    chosen = []
    for r in _rows_from_table(p.get("table")):
        try:
            idx, keep, order = int(r[0]), bool(r[2]), float(r[3])
        except (ValueError, TypeError, IndexError):
            continue
        if keep and 0 <= idx < len(metas):
            chosen.append((order, idx))
    if not chosen:
        chosen = [(i, i) for i in range(len(metas))]
    chosen.sort()

    order_file = work / "order.txt"
    order_file.write_text("\n".join(metas[idx]["name"] for _, idx in chosen))
    out_path = work / "reel.mp4"

    seed_val = p.get("seed")
    cfg = RenderConfig(
        media_dir=media_dir, audio=audio_path, out=out_path, order_file=order_file,
        aspect=ASPECT_LABELS[p["aspect"]],
        style=STYLE_BY_LABEL.get(p["style"], "vivid"),
        cuts_per_beat=SPEEDS[p["speed"]],
        beat_offset=float(p.get("beat_offset") or 0.0),
        xfade_duration=max(0.0, float(p.get("xfade_ms") or 0.0) / 1000.0),
        strong_beats_only=bool(p.get("strong_only")),
        dynamic_pacing=bool(p.get("dynamic_pacing")),
        snap_to_onsets=bool(p.get("snap_onsets")),
        transition_mode=TRANSITIONS.get(p.get("transition"), "auto"),
        edit_style=EDIT_STYLE_BY_LABEL.get(p.get("edit_style"), "clean"),
        motion=MOTION_BY_LABEL.get(p.get("motion"), "kenburns"),
        motion_intensity=float(p.get("motion_intensity") or 1.0),
        impact_effects=tuple(IMPACT_BY_LABEL.get(x, x) for x in (p.get("impact") or [])),
        variety=float(p.get("variety") if p.get("variety") is not None else 0.3),
        grain=float(p.get("grain") or 0.0),
        vignette=bool(p.get("vignette")),
        chromatic=bool(p.get("chromatic")),
        slowmo=bool(p.get("slowmo")),
        hold_prob=float(p.get("hold_prob") or 0.0),
        audio_start=float(p.get("audio_start") or 0.0),
        hook_hold=float(p.get("hook_hold") or 0.0),
        title_text=(p.get("title_text") or "").strip(),
        title_pos=TITLE_POS.get(p.get("title_pos"), "center"),
        title_start=float(p.get("title_start") or 0.0),
        title_duration=float(p.get("title_dur") or 2.5),
        watermark=Path(_as_path(p["wm_file"])) if p.get("wm_file") else None,
        watermark_pos=WM_POS.get(p.get("wm_pos"), "br"),
        watermark_scale=float(p.get("wm_scale") or 0.15),
        watermark_opacity=float(p.get("wm_opacity") or 0.85),
        shuffle=bool(p.get("shuffle")),
        max_duration=float(p["max_dur"]) if p.get("max_dur") else None,
        seed=force_seed if force_seed is not None
        else (int(seed_val) if seed_val not in (None, "") else None),
    )

    try:
        run_pipeline(cfg, progress=lambda fr, msg: progress(fr, desc=msg))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error(f"Errore durante il montaggio: {e}")
    return str(out_path), str(out_path)


with gr.Blocks(title="autoedit — montaggio automatico") as demo:
    state = gr.State()
    gr.Markdown(
        "# 🎬 autoedit\n"
        "**1.** Carica foto/video + una canzone e premi **Analizza**. "
        "**2.** Regola effetti e opzioni, poi **Genera**.\n\n"
        "*Tutto in locale: serve `ffmpeg` installato.*"
    )

    with gr.Row():
        files = gr.File(label="Foto e video", file_count="multiple", file_types=ALLOWED_EXT)
        audio = gr.Audio(label="Canzone (mp3 / wav)", type="filepath")
    analizza_btn = gr.Button("🔍 Analizza", variant="secondary")

    info_md = gr.Markdown()
    wave_img = gr.Image(label="Forma d'onda + beat / downbeat / beat forti", interactive=False)
    gallery = gr.Gallery(label="Media caricati", columns=6, height="auto")
    table = gr.Dataframe(
        headers=TABLE_HEADERS, datatype=["number", "str", "bool", "number"],
        column_count=(4, "fixed"), interactive=True,
        label="Ordine e selezione — cambia «ordine» per riordinare, togli «includi» per escludere",
    )

    gr.Markdown("### Montaggio")
    edit_style = gr.Dropdown(
        list(EDIT_STYLE_BY_LABEL), value=EDIT_STYLE_LABELS["clean"],
        label="Stile di montaggio — ricette coerenti stile TikTok/CapCut (il pannello Effetti serve solo con «Personalizzato»)")
    with gr.Row():
        speed = gr.Dropdown(list(SPEEDS), value=list(SPEEDS)[1], label="Velocità dei tagli")
        aspect = gr.Dropdown(list(ASPECT_LABELS), value=list(ASPECT_LABELS)[0], label="Formato video")
        transition_mode = gr.Dropdown(list(TRANSITIONS), value=list(TRANSITIONS)[0],
                                      label="Transizioni (solo «Personalizzato»)")

    with gr.Accordion("💾 Preset", open=False):
        with gr.Row():
            preset_dd = gr.Dropdown(_list_presets(), label="Preset salvati", scale=3)
            carica_btn = gr.Button("Carica", scale=1)
            elimina_btn = gr.Button("Elimina", variant="stop", scale=1)
        with gr.Row():
            preset_name = gr.Textbox(label="Nome per salvare le impostazioni attuali", scale=3)
            salva_btn = gr.Button("Salva", variant="secondary", scale=1)

    with gr.Accordion("🤖 Brief AI — descrivi il video e lascia scegliere all'LLM", open=False):
        _llm0 = load_llm_config()
        with gr.Row():
            llm_provider = gr.Radio(list(LLM_PROVIDERS),
                                    value=PROVIDER_LABEL.get(_llm0.provider, list(LLM_PROVIDERS)[0]),
                                    label="Provider")
            llm_model = gr.Textbox(value=_llm0.model, label="Modello",
                                   placeholder="es. llama3.1  ·  qwen2.5:7b  ·  claude-opus-5")
        with gr.Row():
            llm_base_url = gr.Textbox(value=_llm0.base_url, label="Base URL (vuoto = default del provider)",
                                      placeholder="Ollama: http://localhost:11434/v1  ·  LM Studio: http://localhost:1234/v1")
            llm_key = gr.Textbox(value=_llm0.api_key, label="API key (le locali spesso non la vogliono)",
                                 type="password")
        llm_save_btn = gr.Button("Salva connessione", variant="secondary")
        brief_text = gr.Textbox(lines=3, label="Brief",
                                placeholder="es. reel energico da spiaggia, taglio veloce, malinconico sul finale, titolo 'ESTATE 2026'")
        brief_btn = gr.Button("✨ Compila impostazioni dal brief", variant="primary")
        llm_status = gr.Markdown()

    with gr.Accordion("🎨 Effetti (solo con stile «Personalizzato»)", open=False):
        with gr.Row():
            style = gr.Dropdown(list(STYLE_BY_LABEL), value=COLOR_LABELS["vivid"],
                                label="Colore / grade di base")
            motion = gr.Dropdown(list(MOTION_BY_LABEL), value=MOTION_LABELS["kenburns"],
                                 label="Movimento")
            motion_intensity = gr.Slider(0.2, 2.5, value=1.0, step=0.1, label="Intensità movimento")
        variety = gr.Slider(0.0, 1.0, value=0.3, step=0.05,
                            label="Varietà — quanto grade/movimento cambiano da clip a clip (anti-monotonia)")
        impact_group = gr.CheckboxGroup(
            list(IMPACT_BY_LABEL), value=[IMPACT_LABELS["flash"], IMPACT_LABELS["rgbsplit"]],
            label="Effetti d'impatto sui beat forti (scelti a caso fra questi)")
        with gr.Row():
            grain = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Grana")
            vignette = gr.Checkbox(value=False, label="Vignettatura")
            chromatic = gr.Checkbox(value=False, label="Aberrazione cromatica")
        with gr.Row():
            slowmo = gr.Checkbox(value=False,
                                 label="Slow-motion + accelerazioni sulle clip video")
            hold_prob = gr.Slider(0.0, 0.4, value=0.0, step=0.02,
                                  label="Clip «hero» tenute 2 beat (0 = mai)")

    with gr.Accordion("✍️ Testo / titolo", open=False):
        title_text = gr.Textbox(label="Testo del titolo (vuoto = niente)", lines=2)
        with gr.Row():
            title_pos = gr.Dropdown(list(TITLE_POS), value="Al centro", label="Posizione")
            title_start = gr.Number(value=0.0, label="Compare al secondo")
            title_dur = gr.Number(value=2.5, label="Resta per (s)")

    with gr.Accordion("🖼️ Logo / watermark", open=False):
        wm_file = gr.Image(label="PNG con trasparenza (vuoto = niente)", type="filepath")
        with gr.Row():
            wm_pos = gr.Dropdown(list(WM_POS), value=list(WM_POS)[0], label="Angolo")
            wm_scale = gr.Slider(0.05, 0.40, value=0.15, step=0.01, label="Dimensione")
            wm_opacity = gr.Slider(0.2, 1.0, value=0.85, step=0.05, label="Opacità")

    with gr.Accordion("⏱️ Ritmo / sincronia / hook", open=False):
        with gr.Row():
            hook_hold = gr.Slider(0.0, 2.0, value=0.0, step=0.1,
                                  label="Hook — tieni fermo il primo clip (s)")
            audio_start = gr.Number(value=0.0, label="Inizio canzone (s) — salta l'intro")
        beat_offset = gr.Slider(-0.20, 0.20, value=0.0, step=0.01,
                                label="Sincronia fine (s) — tagli in ritardo: alza · in anticipo: abbassa")
        xfade_ms = gr.Slider(0, 500, value=180, step=10,
                             label="Durata transizioni (ms) — sotto ~70 = stacchi netti sul beat")
        snap_onsets = gr.Checkbox(value=True, label="Aggancia i tagli ai transienti reali (consigliato)")
        dynamic_pacing = gr.Checkbox(value=False, label="Ritmo dinamico — più tagli nei momenti intensi")
        strong_only = gr.Checkbox(value=False, label="Taglia solo sui beat forti")
        shuffle = gr.Checkbox(value=False, label="Ordine casuale dei media (ignora la tabella)")
        max_dur = gr.Number(label="Durata massima del montaggio (s, vuoto = tutta la canzone)", value=None)
        seed = gr.Number(label="Seed (stesso numero = stesso risultato)", value=None, precision=0)

    with gr.Row():
        genera_btn = gr.Button("🎬 Genera", variant="primary", size="lg")
        variante_btn = gr.Button("🎲 Variante", variant="secondary")
    with gr.Row():
        out_video = gr.Video(label="Anteprima")
        out_file = gr.File(label="Scarica il video")

    def _params(d: dict) -> dict:
        return {
            "table": d[table], "style": d[style], "speed": d[speed], "aspect": d[aspect],
            "edit_style": d[edit_style],
            "transition": d[transition_mode], "motion": d[motion],
            "motion_intensity": d[motion_intensity], "variety": d[variety],
            "impact": d[impact_group], "grain": d[grain], "vignette": d[vignette],
            "chromatic": d[chromatic], "slowmo": d[slowmo], "hold_prob": d[hold_prob],
            "title_text": d[title_text], "title_pos": d[title_pos],
            "title_start": d[title_start], "title_dur": d[title_dur], "wm_file": d[wm_file],
            "wm_pos": d[wm_pos], "wm_scale": d[wm_scale], "wm_opacity": d[wm_opacity],
            "hook_hold": d[hook_hold], "audio_start": d[audio_start], "beat_offset": d[beat_offset],
            "xfade_ms": d[xfade_ms], "snap_onsets": d[snap_onsets], "dynamic_pacing": d[dynamic_pacing],
            "strong_only": d[strong_only], "shuffle": d[shuffle], "max_dur": d[max_dur],
            "seed": d[seed],
        }

    def on_generate(d, progress=gr.Progress()):
        return _do_generate(d[state], _params(d), None, progress)

    def on_variante(d, progress=gr.Progress()):
        return _do_generate(d[state], _params(d), random.randrange(1 << 30), progress)

    # componenti nell'ordine di PRESET_FIELDS (per Carica/Salva)
    PRESET_COMPONENTS = [
        edit_style, style, speed, aspect, transition_mode, motion, motion_intensity, variety,
        impact_group, grain, vignette, chromatic, slowmo, hold_prob, title_text, title_pos,
        title_start, title_dur, wm_pos, wm_scale, wm_opacity, hook_hold, audio_start, beat_offset,
        xfade_ms, snap_onsets, dynamic_pacing, strong_only, shuffle, max_dur, seed,
    ]

    def load_preset(name):
        if not name:
            return [gr.update() for _ in PRESET_FIELDS]
        path = PRESETS_DIR / f"{name}.json"
        if not path.is_file():
            raise gr.Error(f"Preset «{name}» non trovato.")
        data = json.loads(path.read_text())
        return [gr.update(value=data[f]) if f in data else gr.update() for f in PRESET_FIELDS]

    def save_preset(d):
        name = (d[preset_name] or "").strip()
        if not name:
            raise gr.Error("Scrivi un nome per il preset.")
        safe = re.sub(r"[^\w\- ]", "", name).strip() or "preset"
        data = {f: d[c] for f, c in zip(PRESET_FIELDS, PRESET_COMPONENTS)}
        PRESETS_DIR.mkdir(exist_ok=True)
        (PRESETS_DIR / f"{safe}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False))
        return gr.update(choices=_list_presets(), value=safe)

    def delete_preset(name):
        if name:
            (PRESETS_DIR / f"{name}.json").unlink(missing_ok=True)
        return gr.update(choices=_list_presets(), value=None)

    # ---- Brief AI ----
    def _llm_cfg(d) -> LLMConfig:
        return LLMConfig(
            provider=LLM_PROVIDERS.get(d[llm_provider], "openai"),
            base_url=(d[llm_base_url] or "").strip(),
            model=(d[llm_model] or "").strip(),
            api_key=(d[llm_key] or "").strip(),
        )

    def save_llm(d):
        save_llm_config(_llm_cfg(d))
        return "✅ Connessione salvata in `llm_config.json`."

    # componenti aggiornati da un brief, nell'ordine di BRIEF_KEYS
    BRIEF_COMPONENTS = [edit_style, style, motion, motion_intensity, transition_mode, variety,
                        impact_group, grain, vignette, chromatic, slowmo, hold_prob, speed,
                        aspect, strong_only, dynamic_pacing, xfade_ms, hook_hold, title_text,
                        title_pos]
    BRIEF_KEYS = ["edit_style", "style", "motion", "motion_intensity", "transition_mode",
                  "variety", "impact_effects", "grain", "vignette", "chromatic", "slowmo",
                  "hold_prob", "cuts_per_beat", "aspect", "strong_beats_only", "dynamic_pacing",
                  "xfade_ms", "hook_hold", "title_text", "title_pos"]

    def _brief_update(key, ov):
        if key not in ov:
            return gr.update()
        v = ov[key]
        if key == "edit_style":
            return gr.update(value=EDIT_STYLE_LABELS.get(v, v))
        if key == "style":
            return gr.update(value=COLOR_LABELS.get(v, v))
        if key == "motion":
            return gr.update(value=MOTION_LABELS.get(v, v))
        if key == "transition_mode":
            return gr.update(value=TRANS_BY_VALUE.get(v, list(TRANSITIONS)[0]))
        if key == "impact_effects":
            return gr.update(value=[IMPACT_LABELS.get(x, x) for x in v])
        if key == "cuts_per_beat":
            return gr.update(value=SPEED_BY_VALUE.get(v, list(SPEEDS)[1]))
        if key == "aspect":
            return gr.update(value=ASPECT_BY_VALUE.get(v, list(ASPECT_LABELS)[0]))
        if key == "title_pos":
            return gr.update(value=TITLE_BY_VALUE.get(v, "Al centro"))
        return gr.update(value=v)

    def apply_brief(d):
        st = d[state] or {}
        metas = st.get("metas", [])
        beat = st.get("beat", {})
        context = {
            "bpm": beat.get("bpm", "?"), "duration": beat.get("duration", "?"),
            "downbeats": beat.get("downbeats", "?"),
            "n_media": len(metas),
            "n_images": sum(1 for m in metas if m.get("kind") == "image"),
            "n_videos": sum(1 for m in metas if m.get("kind") == "video"),
            "names": [m.get("orig", "") for m in metas],
        }
        try:
            ov = suggest_overrides(_llm_cfg(d), d[brief_text] or "", context)
        except Exception as e:  # noqa: BLE001
            return [gr.update() for _ in BRIEF_COMPONENTS] + [f"❌ {e}"]
        if not ov:
            return [gr.update() for _ in BRIEF_COMPONENTS] + [
                "⚠️ Il modello non ha proposto impostazioni valide. Riprova con un brief più esplicito."]
        applied = ", ".join(f"`{k}`" for k in ov)
        return [_brief_update(k, ov) for k in BRIEF_KEYS] + [f"✅ Applicato: {applied}"]

    gen_set = {
        state, table, style, speed, aspect, edit_style, transition_mode, motion, motion_intensity,
        variety, impact_group, grain, vignette, chromatic, slowmo, hold_prob, title_text,
        title_pos, title_start, title_dur, wm_file, wm_pos, wm_scale, wm_opacity, hook_hold,
        audio_start, beat_offset, xfade_ms, snap_onsets, dynamic_pacing, strong_only, shuffle,
        max_dur, seed,
    }
    analizza_btn.click(_do_analyze, inputs=[files, audio],
                       outputs=[state, gallery, table, wave_img, info_md])
    genera_btn.click(on_generate, inputs=gen_set, outputs=[out_video, out_file])
    variante_btn.click(on_variante, inputs=gen_set, outputs=[out_video, out_file])

    carica_btn.click(load_preset, inputs=[preset_dd], outputs=PRESET_COMPONENTS)
    salva_btn.click(save_preset, inputs=gen_set | {preset_name}, outputs=[preset_dd])
    elimina_btn.click(delete_preset, inputs=[preset_dd], outputs=[preset_dd])

    _llm_set = {llm_provider, llm_model, llm_base_url, llm_key}
    llm_save_btn.click(save_llm, inputs=_llm_set, outputs=[llm_status])
    brief_btn.click(apply_brief, inputs=_llm_set | {state, brief_text},
                    outputs=BRIEF_COMPONENTS + [llm_status])


if __name__ == "__main__":
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ATTENZIONE: 'ffmpeg' / 'ffprobe' non trovati nel PATH. "
              "Installali (es. 'brew install ffmpeg') o il montaggio fallira'.\n")
    demo.launch(inbrowser=True)
