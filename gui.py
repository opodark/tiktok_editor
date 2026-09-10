#!/usr/bin/env python3
"""Interfaccia grafica per autoedit (gira nel browser, tutto in locale).

Avvio:
    pip install -r requirements-gui.txt
    python gui.py

Flusso:
  1. MEDIA & MUSICA  -> carichi foto/video + canzone, vedi miniature, forma
     d'onda con i beat, e decidi l'inquadratura (ritaglio) con l'anteprima.
  2. MONTAGGIO       -> stile, ritmo, effetti; eventualmente un brief all'LLM.
  3. GRAFICA & TESTO -> titolo e logo.
  4. GENERA          -> monti il video (o una variante) e lo scarichi.
  ⚙️ IMPOSTAZIONI    -> connessione LLM, ffmpeg, file temporanei.
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

from autoedit.assets import (
    TEMPLATES, AssetSpec, capabilities as asset_capabilities, render_asset, validate_asset_spec,
)
from autoedit.beats import detect_beats
from autoedit.brief import (
    LLMConfig, load_llm_config, save_llm_config, suggest_asset, suggest_overrides,
)
from autoedit.effects import COLOR_LABELS, EDIT_STYLE_LABELS, IMPACT_LABELS, MOTION_LABELS
from autoedit.media import IMAGE_EXT, VIDEO_EXT
from autoedit.pipeline import ASPECTS, RenderConfig, run_pipeline

from dataclasses import asdict as _asdict

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
    "shuffle", "max_dur", "seed", "crop_zoom", "crop_x", "crop_y",
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
             "-vf", "scale=480:-2", str(out), "-loglevel", "error"],
            check=False,
        )
    if not out.exists():
        try:
            im = Image.open(path)
            im.thumbnail((640, 1138))
            im.convert("RGB").save(out, "JPEG", quality=88)
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


# --- anteprima ritaglio (stessa matematica di effects.crop_to_fill, in PIL) --
def _crop_preview(src: Path, aspect: str, zoom: float, cx: float, cy: float,
                   box_h: int = 760) -> tuple[Image.Image, Image.Image]:
    """Ritorna (frame_risultante, originale_con_riquadro).

    `aspect` e' una chiave di ASPECTS ("9:16"...). `zoom` >= 1 stringe,
    `cx`/`cy` in [0,1] spostano l'inquadratura (0.5 = centro).
    """
    tw, th = ASPECTS[aspect]
    ar = tw / th
    box_w = int(round(box_h * ar))
    z = max(1.0, float(zoom))
    cx = min(1.0, max(0.0, float(cx)))
    cy = min(1.0, max(0.0, float(cy)))

    im = Image.open(src).convert("RGB")
    W, H = im.size
    # scala per RIEMPIRE box_w x box_h (lato corto copre), poi zoom extra
    s = max(box_w / W, box_h / H) * z
    sw, sh = max(box_w, int(round(W * s))), max(box_h, int(round(H * s)))
    scaled = im.resize((sw, sh), Image.LANCZOS)

    x0 = int(round((sw - box_w) * cx))
    y0 = int(round((sh - box_h) * cy))
    result = scaled.crop((x0, y0, x0 + box_w, y0 + box_h))

    # originale (ridotto) con il rettangolo di taglio sovrapposto
    prev_w = 460
    prev_h = max(1, int(round(H * prev_w / W)))
    orig = im.resize((prev_w, prev_h), Image.LANCZOS).convert("RGBA")
    k = prev_w / sw
    rx0, ry0 = x0 * k, y0 * k
    rx1, ry1 = (x0 + box_w) * k, (y0 + box_h) * k
    shade = Image.new("RGBA", orig.size, (0, 0, 0, 0))
    ds = ImageDraw.Draw(shade)
    ds.rectangle((0, 0, prev_w, prev_h), fill=(0, 0, 0, 110))
    ds.rectangle((rx0, ry0, rx1, ry1), fill=(0, 0, 0, 0))
    ds.rectangle((rx0, ry0, rx1, ry1), outline=(255, 0, 80, 255), width=3)
    orig = Image.alpha_composite(orig, shade)
    return result, orig.convert("RGB")


def _preview_from_state(state, which, aspect_label, zoom, cx, cy):
    if not state or not state.get("metas"):
        return None, None
    metas = state["metas"]
    idx = 0
    if which:
        m = re.match(r"\s*(\d+)", str(which))
        if m:
            idx = min(len(metas) - 1, max(0, int(m.group(1))))
    src = Path(metas[idx]["thumb"])
    aspect = ASPECT_LABELS.get(aspect_label, "9:16")
    try:
        return _crop_preview(src, aspect, zoom, cx, cy)
    except Exception:  # noqa: BLE001
        return None, None


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

    from autoedit.palette import describe as _describe_palette
    palette = _describe_palette([m["thumb"] for m in metas], k=4)

    gallery = [(m["thumb"], f'{i} · {m["orig"]}') for i, m in enumerate(metas)]
    table = [[i, m["orig"], True, i] for i, m in enumerate(metas)]
    info = (
        f"### Brano · {bi.tempo:.0f} BPM\n"
        f"- {len(bi.beat_times)} beat · {len(bi.downbeat_times)} downbeat (gialli) · "
        f"{len(bi.strong_times)} beat forti (rossi) · {len(bi.onset_times)} transienti\n"
        f"- durata {bi.duration:.1f}s · **{len(metas)} media** caricati\n"
        f"- palette: {palette or 'n/d'}"
    )
    state = {"work": str(work), "media_dir": str(media_dir),
             "audio": str(audio_path), "metas": metas, "palette": palette,
             "beat": {"bpm": round(float(bi.tempo)), "duration": round(float(bi.duration), 1),
                      "downbeats": int(len(bi.downbeat_times))}}
    which_choices = [f'{i} · {m["orig"]}' for i, m in enumerate(metas)]
    prev_res, prev_orig = (None, None)
    try:
        prev_res, prev_orig = _crop_preview(Path(metas[0]["thumb"]), "9:16", 1.0, 0.5, 0.5)
    except Exception:  # noqa: BLE001
        pass
    return (state, gallery, table, str(wave), info,
            gr.update(choices=which_choices, value=which_choices[0] if which_choices else None),
            prev_res, prev_orig)


def _do_generate(state, p: dict, force_seed, progress) -> tuple[str, str]:
    if not state:
        raise gr.Error("Premi prima «Analizza» nella tab Media & Musica.")
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
        fps=int(p.get("fps") or 30),
        crop_zoom=float(p.get("crop_zoom") or 1.0),
        crop_x=float(p.get("crop_x") if p.get("crop_x") is not None else 0.5),
        crop_y=float(p.get("crop_y") if p.get("crop_y") is not None else 0.5),
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
        overlay_specs=tuple(p.get("overlay_specs") or ()),
        shuffle=bool(p.get("shuffle")),
        max_duration=float(p["max_dur"]) if p.get("max_dur") else None,
        jobs=int(p.get("jobs") or 0),
        chunk_size=int(p.get("chunk_size") or 10),
        keep_temp=bool(p.get("keep_temp")),
        seed=force_seed if force_seed is not None
        else (int(seed_val) if seed_val not in (None, "") else None),
    )

    try:
        run_pipeline(cfg, progress=lambda fr, msg: progress(fr, desc=msg))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error(f"Errore durante il montaggio: {e}")
    return str(out_path), str(out_path)


CSS = """
.hint { font-size: 12px; opacity: .7; margin: -6px 0 8px; }
"""

with gr.Blocks(title="autoedit — montaggio automatico") as demo:
    state = gr.State()
    gr.Markdown("# 🎬 autoedit\n*Montaggio verticale sincronizzato sul beat. Tutto in locale.*")

    with gr.Tabs():
        # =================================================================
        # 1 · MEDIA & MUSICA
        # =================================================================
        with gr.Tab("1 · Media & Musica"):
            with gr.Row():
                files = gr.File(label="Foto e video", file_count="multiple", file_types=ALLOWED_EXT)
                audio = gr.Audio(label="Canzone (mp3 / wav)", type="filepath")
            gr.Markdown(
                "<div class='hint'>La canzone serve solo a trovare il beat: puoi montare "
                "su un mp3 e poi rimettere la traccia dalla libreria di TikTok (stessa canzone).</div>")
            analizza_btn = gr.Button("🔍 Analizza", variant="primary")

            info_md = gr.Markdown()
            wave_img = gr.Image(label="Forma d'onda + beat / downbeat (giallo) / beat forti (rosso)",
                                interactive=False)
            gallery = gr.Gallery(label="Media caricati", columns=6, height="auto")
            table = gr.Dataframe(
                headers=TABLE_HEADERS, datatype=["number", "str", "bool", "number"],
                column_count=(4, "fixed"), interactive=True,
                label="Ordine e selezione",
            )
            gr.Markdown("<div class='hint'>Cambia la colonna «ordine» per riordinare · "
                        "togli la spunta «includi» per escludere una clip.</div>")

            with gr.Accordion("🖼️ Inquadratura / ritaglio", open=True):
                gr.Markdown(
                    "<div class='hint'>Come vengono ritagliate le clip per riempire il formato. "
                    "Vale per tutte le clip. A sinistra il risultato, a destra dov'e' il taglio "
                    "sull'originale.</div>")
                with gr.Row():
                    crop_preview_out = gr.Image(label="Come apparira'", interactive=False, height=380)
                    crop_orig_out = gr.Image(label="Taglio sull'originale", interactive=False, height=380)
                crop_which = gr.Dropdown(
                    [], label="Anteprima su quale media",
                    info="Scegli una clip caricata; l'inquadratura poi vale per tutte.")
                with gr.Row():
                    crop_zoom = gr.Slider(1.0, 2.5, value=1.0, step=0.05, label="Zoom",
                                          info="1.0 = riempi e basta. Più alto = inquadratura più stretta.")
                    crop_x = gr.Slider(0.0, 1.0, value=0.5, step=0.02, label="Orizzontale",
                                       info="0 = verso sinistra · 0.5 = centro · 1 = verso destra.")
                    crop_y = gr.Slider(0.0, 1.0, value=0.5, step=0.02, label="Verticale",
                                       info="0 = verso l'alto · 0.5 = centro · 1 = verso il basso. "
                                            "Foto da telefono: spesso 0.35–0.45 tiene i volti in campo.")

        # =================================================================
        # 2 · MONTAGGIO
        # =================================================================
        with gr.Tab("2 · Montaggio"):
            edit_style = gr.Dropdown(
                list(EDIT_STYLE_BY_LABEL), value=EDIT_STYLE_LABELS["clean"],
                label="Stile di montaggio",
                info="Ricette coerenti stile TikTok/CapCut. I pannelli Effetti/Transizioni "
                     "contano solo con «Personalizzato».")
            with gr.Row():
                speed = gr.Dropdown(list(SPEEDS), value=list(SPEEDS)[1], label="Velocità dei tagli",
                                    info="Quanti stacchi per battito musicale.")
                aspect = gr.Dropdown(list(ASPECT_LABELS), value=list(ASPECT_LABELS)[0],
                                     label="Formato video",
                                     info="9:16 per TikTok/Reels/Shorts. Cambia anche l'anteprima ritaglio.")
                transition_mode = gr.Dropdown(list(TRANSITIONS), value=list(TRANSITIONS)[0],
                                              label="Transizioni",
                                              info="Solo con stile «Personalizzato».")

            with gr.Accordion("🤖 Brief AI — descrivi il video, sceglie l'LLM", open=False):
                gr.Markdown("<div class='hint'>La connessione al modello si imposta nella tab "
                            "⚙️ Impostazioni. Qui scrivi cosa vuoi e premi Compila.</div>")
                brief_text = gr.Textbox(
                    lines=3, label="Brief",
                    placeholder="es. reel energico da spiaggia, taglio veloce, malinconico sul finale, "
                                "titolo 'ESTATE 2026'")
                brief_btn = gr.Button("✨ Compila impostazioni dal brief", variant="primary")
                llm_status = gr.Markdown()

            with gr.Accordion("🎨 Effetti (solo con stile «Personalizzato»)", open=False):
                with gr.Row():
                    style = gr.Dropdown(list(STYLE_BY_LABEL), value=COLOR_LABELS["vivid"],
                                        label="Colore / grade di base",
                                        info="Correzione colore applicata a tutte le clip.")
                    motion = gr.Dropdown(list(MOTION_BY_LABEL), value=MOTION_LABELS["kenburns"],
                                         label="Movimento",
                                         info="Come si muove l'inquadratura dentro ogni clip.")
                    motion_intensity = gr.Slider(0.2, 2.5, value=1.0, step=0.1,
                                                 label="Intensità movimento",
                                                 info="Ampiezza di zoom e oscillazioni.")
                variety = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Varietà",
                                    info="Quanto grade e movimento cambiano da clip a clip. "
                                         "Alto = anti-monotonia, basso = uniforme.")
                impact_group = gr.CheckboxGroup(
                    list(IMPACT_BY_LABEL), value=[IMPACT_LABELS["flash"], IMPACT_LABELS["rgbsplit"]],
                    label="Effetti d'impatto sui beat forti",
                    info="Ne viene scelto uno a caso fra quelli spuntati, sui colpi forti.")
                with gr.Row():
                    grain = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Grana",
                                      info="Rumore tipo pellicola.")
                    vignette = gr.Checkbox(value=False, label="Vignettatura",
                                           info="Bordi leggermente più scuri.")
                    chromatic = gr.Checkbox(value=False, label="Aberrazione cromatica",
                                            info="Sfrangiatura RGB leggera su tutto.")
                with gr.Row():
                    slowmo = gr.Checkbox(value=False, label="Slow-motion + accelerazioni (clip video)",
                                         info="Rallenta/velocizza gli spezzoni video a ritmo.")
                    hold_prob = gr.Slider(0.0, 0.4, value=0.0, step=0.02,
                                          label="Clip «hero» tenute 2 beat",
                                          info="Probabilità che una clip resti in campo il doppio. 0 = mai.")

            with gr.Accordion("⏱️ Ritmo / sincronia / hook", open=False):
                with gr.Row():
                    hook_hold = gr.Slider(0.0, 2.0, value=0.0, step=0.1,
                                          label="Hook — primo clip fermo (s)",
                                          info="Tiene fermo il primo clip N secondi prima che parta il "
                                               "montaggio. Il primo taglio resta sul beat.")
                    audio_start = gr.Number(value=0.0, label="Inizio canzone (s)",
                                            info="Salta l'intro del brano.")
                beat_offset = gr.Slider(-0.20, 0.20, value=0.0, step=0.01, label="Sincronia fine (s)",
                                        info="Se i tagli «sentono» in ritardo alza, se in anticipo abbassa.")
                xfade_ms = gr.Slider(0, 500, value=180, step=10, label="Durata transizioni (ms)",
                                     info="Sotto ~70 = stacchi netti sul beat.")
                snap_onsets = gr.Checkbox(value=True, label="Aggancia i tagli ai transienti reali",
                                          info="Consigliato: i sotto-tagli cadono su attacchi veri del brano.")
                dynamic_pacing = gr.Checkbox(value=False, label="Ritmo dinamico",
                                             info="Più tagli nei tratti intensi, meno nei tratti calmi.")
                strong_only = gr.Checkbox(value=False, label="Taglia solo sui beat forti",
                                          info="Montaggio più calmo.")
                shuffle = gr.Checkbox(value=False, label="Ordine casuale dei media",
                                      info="Ignora la colonna «ordine» della tabella.")
                with gr.Row():
                    max_dur = gr.Number(label="Durata massima (s)", value=None,
                                        info="Vuoto = tutta la canzone.")
                    seed = gr.Number(label="Seed", value=None, precision=0,
                                     info="Stesso numero = stesso montaggio. Vuoto = casuale.")

        # =================================================================
        # 3 · GRAFICA & TESTO
        # =================================================================
        with gr.Tab("3 · Grafica & Testo"):
            with gr.Accordion("✍️ Titolo in sovrimpressione", open=True):
                title_text = gr.Textbox(label="Testo del titolo", lines=2,
                                        info="Vuoto = nessun titolo. A capo automatico.")
                with gr.Row():
                    title_pos = gr.Dropdown(list(TITLE_POS), value="Al centro", label="Posizione")
                    title_start = gr.Number(value=0.0, label="Compare al secondo")
                    title_dur = gr.Number(value=2.5, label="Resta per (s)")

            with gr.Accordion("🖼️ Logo / watermark", open=False):
                wm_file = gr.Image(label="PNG con trasparenza (vuoto = nessun logo)",
                                   type="filepath")
                with gr.Row():
                    wm_pos = gr.Dropdown(list(WM_POS), value=list(WM_POS)[0], label="Angolo")
                    wm_scale = gr.Slider(0.05, 0.40, value=0.15, step=0.01, label="Dimensione",
                                         info="Larghezza come frazione del video.")
                    wm_opacity = gr.Slider(0.2, 1.0, value=0.85, step=0.05, label="Opacità")

            with gr.Accordion("✨ Grafica generata (LLM)", open=False):
                gr.Markdown(
                    "<div class='hint'>Descrivi un elemento grafico (logo, card titolo, badge, "
                    "@handle, prezzo, CTA…). Il modello grafica lo genera come PNG trasparente; "
                    "puoi ritoccare il JSON e rigenerare, poi aggiungerlo al montaggio.</div>")
                asset_brief = gr.Textbox(
                    lines=2, label="Cosa generare",
                    placeholder="es. badge rosso «RUBATA» · titolo «Come mi hanno rubato la moto» · "
                                "wordmark «MOTO LIFE» con LIFE rosa · @opodark in basso")
                asset_kind = gr.Radio(
                    ["Auto", "Template", "SVG", "Maschera testo"], value="Auto",
                    label="Tipo", info="Auto = sceglie il modello. «Maschera testo» = il video "
                                       "scorrerà dentro le lettere.")
                asset_gen_btn = gr.Button("✨ Genera grafica", variant="primary")
                asset_status = gr.Markdown()
                with gr.Row():
                    asset_preview = gr.Image(label="Anteprima PNG", interactive=False, height=300)
                    asset_spec_box = gr.Code(label="AssetSpec (JSON) — modificabile", language="json")
                asset_rerender_btn = gr.Button("↻ Rigenera dal JSON")

                gr.Markdown("<div class='hint'>Posizionamento nel video:</div>")
                with gr.Row():
                    asset_pos = gr.Dropdown(
                        ["center", "top", "bottom", "left", "right", "tl", "tr", "bl", "br"],
                        value="center", label="Posizione")
                    asset_scale = gr.Slider(0.05, 1.0, value=0.6, step=0.02,
                                            label="Dimensione", info="Larghezza come frazione del video.")
                    asset_opacity = gr.Slider(0.2, 1.0, value=1.0, step=0.05, label="Opacità")
                with gr.Row():
                    asset_timed = gr.Checkbox(value=False, label="Solo per un tratto",
                                              info="Altrimenti resta per tutto il video.")
                    asset_start = gr.Number(value=0.0, label="Da (s)")
                    asset_dur = gr.Number(value=3.0, label="Durata (s)")
                with gr.Row():
                    asset_add_btn = gr.Button("➕ Aggiungi al montaggio", variant="secondary")
                    asset_clear_btn = gr.Button("🗑️ Svuota", variant="stop")
                asset_queue_md = gr.Markdown("*Nessuna grafica aggiunta.*")
                asset_queue = gr.State([])
                asset_cur = gr.State(None)

        # =================================================================
        # 4 · GENERA
        # =================================================================
        with gr.Tab("4 · Genera"):
            with gr.Row():
                genera_btn = gr.Button("🎬 Genera", variant="primary", size="lg")
                variante_btn = gr.Button("🎲 Variante", variant="secondary",
                                         size="lg")
            gr.Markdown("<div class='hint'>«Variante» rimonta la stessa musica e gli stessi media "
                        "con un seed diverso.</div>")
            with gr.Row():
                out_video = gr.Video(label="Anteprima")
                out_file = gr.File(label="Scarica il video")

            with gr.Accordion("💾 Preset", open=False):
                gr.Markdown("<div class='hint'>Salva/riusa tutte le impostazioni di montaggio, "
                            "grafica e inquadratura (non i media).</div>")
                with gr.Row():
                    preset_dd = gr.Dropdown(_list_presets(), label="Preset salvati", scale=3)
                    carica_btn = gr.Button("Carica", scale=1)
                    elimina_btn = gr.Button("Elimina", variant="stop", scale=1)
                with gr.Row():
                    preset_name = gr.Textbox(label="Nome per salvare le impostazioni attuali", scale=3)
                    salva_btn = gr.Button("Salva", variant="secondary", scale=1)

        # =================================================================
        # ⚙️ IMPOSTAZIONI
        # =================================================================
        with gr.Tab("⚙️ Impostazioni"):
            _llm0 = load_llm_config()
            with gr.Accordion("🤖 Connessione LLM (per il Brief AI)", open=True):
                with gr.Row():
                    llm_provider = gr.Radio(
                        list(LLM_PROVIDERS),
                        value=PROVIDER_LABEL.get(_llm0.provider, list(LLM_PROVIDERS)[0]),
                        label="Provider",
                        info="«Compatibile OpenAI» copre Ollama, LM Studio, llama.cpp, vLLM…")
                    llm_model = gr.Textbox(
                        value=_llm0.model, label="Modello (brief montaggio)",
                        placeholder="qwen2.5:7b  ·  hf.co/utente/repo:Q4_K_M  ·  claude-sonnet-5",
                        info="Nome esatto del modello nel provider.")
                with gr.Row():
                    llm_asset_model = gr.Textbox(
                        value=_llm0.asset_model, label="Modello per la grafica (vuoto = come sopra)",
                        placeholder="qwen2.5-coder:14b",
                        info="Un modello 'coder' fa SVG e layout più puliti.")
                    llm_base_url = gr.Textbox(
                        value=_llm0.base_url, label="Base URL",
                        placeholder="http://localhost:11434/v1",
                        info="Vuoto = default del provider. Ollama: http://localhost:11434/v1")
                llm_key = gr.Textbox(value=_llm0.api_key, label="API key", type="password",
                                     info="I modelli locali di solito non la richiedono.")
                llm_save_btn = gr.Button("Salva connessione", variant="secondary")
                llm_cfg_status = gr.Markdown()

            with gr.Accordion("⚙️ Rendering (ffmpeg)", open=False):
                with gr.Row():
                    fps_set = gr.Slider(24, 60, value=30, step=1, label="FPS di output",
                                        info="30 va bene per TikTok. 60 = più fluido, file più pesante.")
                    jobs_set = gr.Slider(0, 16, value=0, step=1, label="Processi ffmpeg paralleli",
                                         info="0 = automatico (in base ai core della CPU).")
                    chunk_set = gr.Slider(4, 24, value=10, step=1, label="Segmenti per catena",
                                          info="Più alto = meno file intermedi ma più RAM. Se il "
                                               "montaggio viene ucciso a metà, abbassa.")
                keep_temp = gr.Checkbox(value=False, label="Tieni i file temporanei",
                                        info="Per debug: non cancella la cartella di lavoro.")

            with gr.Accordion("🐞 DEBUG — vista mirino IA", open=True):
                gr.Markdown(
                    "<div class='hint'>Guardi la clip attraverso un mirino da reflex: le staffe "
                    "AF scattano su ciò che l'IA riconosce (busto / presa), con palo, scheletro, "
                    "timeline dei «fermi» e HUD. Serve per capire cosa sta capendo l'IA.</div>")
                dbg_video_in = gr.Video(label="Clip da analizzare")
                dbg_video_path = gr.Textbox(
                    label="…oppure incolla il percorso del file (se l'upload dà «errore video»)",
                    placeholder=r"D:\video\allenamento_palo.mp4",
                    info="Il file deve essere già esportato e chiuso dall'editor. "
                         "Meglio se in una cartella normale (Desktop, Video), non in Temp.")
                with gr.Row():
                    dbg_pole_x = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
                                           label="Palo — x manuale (0 = auto)",
                                           info="Se l'auto non lo trova: metti dove sta il palo "
                                                "in orizzontale (0=sx, 0.5=centro, 1=dx).")
                    dbg_still = gr.Slider(0.004, 0.05, value=0.012, step=0.002,
                                          label="Soglia «fermo»",
                                          info="Più alta = più tollerante (trova più fermi). "
                                               "Se ne trova troppi, abbassala.")
                    dbg_minhold = gr.Slider(0.3, 2.0, value=0.6, step=0.1,
                                            label="Durata minima fermo (s)")
                dbg_events = gr.Checkbox(
                    value=True, label="Segna inversioni ed estensioni massime",
                    info="Corpo a testa in giù · braccio/gamba che raggiunge l'estensione piena · "
                         "apertura massima delle gambe. Geometria pura dei keypoint, niente LLM.")
                with gr.Row():
                    dbg_use_vlm = gr.Checkbox(
                        value=False, label="Usa anche il modello visione sui fermi",
                        info="Nomina la mossa + la presa su ogni fermo. Più lento (~6s/fermo).")
                    dbg_moves = gr.Textbox(
                        label="Lista mosse per name_pose (una per riga)", lines=2,
                        placeholder="Basic invert\nSuperman\nGemini\nAyesha…")
                dbg_btn = gr.Button("🐞  GENERA VIDEO DEBUG", variant="stop", size="lg")
                dbg_out = gr.Video(label="Video mirino")
                dbg_log = gr.Markdown()

    # ---------------------------------------------------------------------
    # Wiring
    # ---------------------------------------------------------------------
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
            "crop_zoom": d[crop_zoom], "crop_x": d[crop_x], "crop_y": d[crop_y],
            "fps": d[fps_set], "jobs": d[jobs_set], "chunk_size": d[chunk_set],
            "keep_temp": d[keep_temp],
            "overlay_specs": d[asset_queue],
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
        crop_zoom, crop_x, crop_y,
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

    # ---- LLM / Brief ----
    def _llm_cfg(d) -> LLMConfig:
        return LLMConfig(
            provider=LLM_PROVIDERS.get(d[llm_provider], "openai"),
            base_url=(d[llm_base_url] or "").strip(),
            model=(d[llm_model] or "").strip(),
            asset_model=(d[llm_asset_model] or "").strip(),
            api_key=(d[llm_key] or "").strip(),
        )

    def save_llm(d):
        save_llm_config(_llm_cfg(d))
        return "✅ Connessione salvata in `llm_config.json`."

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
            "palette": st.get("palette", ""),
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

    # ---- grafica generata (LLM -> AssetSpec -> PNG) ----
    _KIND_HINT = {"Template": "\n(genera come 'template')", "SVG": "\n(genera come kind='svg')",
                  "Maschera testo": "\n(genera come kind='text_mask')", "Auto": ""}

    def _spec_to_preview(spec: AssetSpec):
        d = Path(tempfile.mkdtemp(prefix="autoedit_asset_"))
        png = render_asset(spec, d)
        return str(png), {"png": str(png), "spec": _asdict(spec)}

    def _queue_md(q):
        if not q:
            return "*Nessuna grafica aggiunta.*"
        rows = []
        for i, x in enumerate(q):
            line = f"{i + 1}. `{Path(x['png']).name}` · {x['pos']} · scala {x['scale']:.2f}"
            if x.get("dur"):
                line += f" · {x['start']:.1f}–{x['start'] + x['dur']:.1f}s"
            rows.append(line)
        return "**In coda:**\n" + "\n".join(rows)

    def on_asset_generate(d):
        brief = (d[asset_brief] or "").strip()
        if not brief:
            return gr.update(), gr.update(), "⚠️ Scrivi cosa vuoi generare.", gr.update()
        st = d[state] or {}
        metas = st.get("metas", [])
        ctx = {"aspect": ASPECT_LABELS.get(d[aspect], "9:16")}
        if st.get("palette"):
            ctx["palette"] = st["palette"]
        if metas:
            ctx["names"] = [m.get("orig", "") for m in metas][:30]
            ctx["mood"] = (f"{len(metas)} media "
                           f"({sum(1 for m in metas if m.get('kind') == 'image')} foto, "
                           f"{sum(1 for m in metas if m.get('kind') == 'video')} video)")
        try:
            spec = suggest_asset(_llm_cfg(d), brief + _KIND_HINT.get(d[asset_kind], ""), ctx)
            png, cur = _spec_to_preview(spec)
        except Exception as e:  # noqa: BLE001
            return gr.update(), gr.update(), f"❌ {e}", gr.update()
        js = json.dumps(_asdict(spec), indent=2, ensure_ascii=False)
        return png, js, f"✅ {spec.kind} · {spec.template or spec.asset_id}", cur

    def on_asset_rerender(js):
        try:
            spec = validate_asset_spec(json.loads(js or "{}"))
            png, cur = _spec_to_preview(spec)
        except Exception as e:  # noqa: BLE001
            return gr.update(), f"❌ {e}", gr.update()
        return png, f"✅ rigenerato · {spec.kind}", cur

    def on_asset_add(cur, pos, scale, opacity, timed, start, dur, q):
        if not cur or not cur.get("png"):
            return q, "⚠️ Genera prima una grafica."
        q = list(q or [])
        item = {"png": cur["png"], "pos": pos,
                "scale": float(scale), "opacity": float(opacity)}
        if timed:
            item["start"] = float(start or 0.0)
            item["dur"] = float(dur or 3.0)
        q.append(item)
        return q, _queue_md(q)

    def on_asset_clear():
        return [], _queue_md([])

    _asset_gen_in = {state, asset_brief, asset_kind, aspect, llm_provider, llm_model,
                     llm_asset_model, llm_base_url, llm_key}
    asset_gen_btn.click(on_asset_generate, inputs=_asset_gen_in,
                        outputs=[asset_preview, asset_spec_box, asset_status, asset_cur])
    asset_rerender_btn.click(on_asset_rerender, inputs=[asset_spec_box],
                             outputs=[asset_preview, asset_status, asset_cur])
    asset_add_btn.click(
        on_asset_add,
        inputs=[asset_cur, asset_pos, asset_scale, asset_opacity, asset_timed,
                asset_start, asset_dur, asset_queue],
        outputs=[asset_queue, asset_queue_md])
    asset_clear_btn.click(on_asset_clear, outputs=[asset_queue, asset_queue_md])

    # ---- DEBUG: vista mirino IA ----
    def on_debug(d, progress=gr.Progress()):
        video = (d[dbg_video_path] or "").strip().strip('"') or d[dbg_video_in]
        if not video:
            raise gr.Error("Carica una clip (o incolla il percorso) nella sezione DEBUG.")
        try:
            from autoedit import pose, vision
        except Exception as e:  # noqa: BLE001
            raise gr.Error(f"Manca una dipendenza: {e}. Installa:  pip install \"autoedit[pose]\"")
        vpath = _as_path(video)
        if not vpath.is_file():
            raise gr.Error(f"File non trovato o non leggibile: {vpath}. "
                           "Assicurati che l'export sia finito e il file chiuso.")
        try:
            with open(vpath, "rb") as _f:
                _f.read(1024)
        except Exception as e:  # noqa: BLE001
            raise gr.Error(f"Il file è bloccato da un altro programma ({e}). "
                           "Chiudi l'editor / attendi la fine dell'export, o copialo altrove.")
        progress(0.1, desc="Analisi pose (MediaPipe)…")
        frames = pose.analyze_video(vpath, fps_sample=8)
        seen = sum(1 for f in frames if f.lm is not None)
        manual = float(d[dbg_pole_x] or 0.0)
        if manual > 0.0:
            px, pole_src = manual, "manuale"
        else:
            px, pole_src = pose.pole_x_auto(vpath, frames)
        cts = pose.contacts(frames, px)
        holds = pose.detect_holds(frames, cts, still=float(d[dbg_still]),
                                  min_hold=float(d[dbg_minhold]))
        events = pose.detect_events(frames) if d[dbg_events] else []

        labels: dict = {}
        if d[dbg_use_vlm] and holds:
            moves = [m.strip() for m in (d[dbg_moves] or "").splitlines() if m.strip()]
            cfg = _llm_cfg(d)
            tmpd = Path(tempfile.mkdtemp(prefix="autoedit_dbgf_"))
            for k, hd in enumerate(holds):
                progress(0.25 + 0.55 * k / len(holds), desc=f"Modello visione · fermo {k + 1}/{len(holds)}")
                fp = tmpd / f"h{k}.png"
                try:
                    pose.save_frame(vpath, hd.focus_t, fp)
                    gp = vision.grip_part(cfg, fp)
                    nm = vision.name_pose(cfg, fp, moves)
                    labels[k] = {"grip": gp.get("load_bearing") or ", ".join(gp.get("parts") or []) or "?",
                                 "move": nm.get("move", "")}
                except Exception as e:  # noqa: BLE001
                    labels[k] = {"grip": f"err: {e}", "move": ""}

        progress(0.85, desc="Rendering mirino…")
        out = Path(tempfile.mkdtemp(prefix="autoedit_dbg_")) / "debug.mp4"
        pose.debug_video(vpath, out, frames, holds, cts, px, labels, events=events)

        lines = [f"**Frame campionati:** {len(frames)} · persona rilevata in **{seen}**",
                 (f"**Palo:** x={px:.3f} ({pole_src})" if px is not None
                  else "**Palo:** non trovato — mettilo a mano con lo slider «Palo — x manuale»"),
                 f"**Contatti:** {len(cts)} · **Fermi:** {len(holds)}"]
        if px is None:
            lines.append("_Senza palo non ci sono contatti/prese: le staffe AF seguono solo il busto._")
        if not holds:
            lines.append("_0 fermi: la ballerina si muove sempre o il jitter supera la soglia — "
                         "alza «Soglia fermo» e/o abbassa «Durata minima»._")
        if events:
            lines.append(f"**Eventi ({len(events)}):**")
            for e in events:
                lines.append(f"- `{e.t:.1f}s` · {e.kind} · {e.part or '—'} · {e.value:.0f}° · {e.label}")
        for k, hd in enumerate(holds):
            lab = labels.get(k, {})
            lines.append(
                f"- fermo `{hd.t0:.1f}–{hd.t1:.1f}s` · focus@{hd.focus_t:.1f}s · "
                f"presa: {lab.get('grip') or hd.focus_part or '—'} · mossa: {lab.get('move') or '—'}")
        return str(out), "\n".join(lines)

    dbg_btn.click(
        on_debug,
        inputs={dbg_video_in, dbg_video_path, dbg_pole_x, dbg_still, dbg_minhold, dbg_events,
                dbg_use_vlm, dbg_moves, llm_provider, llm_model, llm_asset_model,
                llm_base_url, llm_key},
        outputs=[dbg_out, dbg_log])

    # ---- anteprima ritaglio ----
    _crop_inputs = [state, crop_which, aspect, crop_zoom, crop_x, crop_y]
    _crop_outputs = [crop_preview_out, crop_orig_out]
    for _c in (crop_which, aspect):
        _c.change(_preview_from_state, inputs=_crop_inputs, outputs=_crop_outputs)
    for _c in (crop_zoom, crop_x, crop_y):
        _c.release(_preview_from_state, inputs=_crop_inputs, outputs=_crop_outputs)

    gen_set = {
        state, table, style, speed, aspect, edit_style, transition_mode, motion, motion_intensity,
        variety, impact_group, grain, vignette, chromatic, slowmo, hold_prob, title_text,
        title_pos, title_start, title_dur, wm_file, wm_pos, wm_scale, wm_opacity, hook_hold,
        audio_start, beat_offset, xfade_ms, snap_onsets, dynamic_pacing, strong_only, shuffle,
        max_dur, seed, crop_zoom, crop_x, crop_y, fps_set, jobs_set, chunk_set, keep_temp,
        asset_queue,
    }
    analizza_btn.click(
        _do_analyze, inputs=[files, audio],
        outputs=[state, gallery, table, wave_img, info_md, crop_which,
                 crop_preview_out, crop_orig_out])
    genera_btn.click(on_generate, inputs=gen_set, outputs=[out_video, out_file])
    variante_btn.click(on_variante, inputs=gen_set, outputs=[out_video, out_file])

    carica_btn.click(load_preset, inputs=[preset_dd], outputs=PRESET_COMPONENTS)
    salva_btn.click(save_preset, inputs=gen_set | {preset_name}, outputs=[preset_dd])
    elimina_btn.click(delete_preset, inputs=[preset_dd], outputs=[preset_dd])

    _llm_set = {llm_provider, llm_model, llm_asset_model, llm_base_url, llm_key}
    llm_save_btn.click(save_llm, inputs=_llm_set, outputs=[llm_cfg_status])
    brief_btn.click(apply_brief, inputs=_llm_set | {state, brief_text},
                    outputs=BRIEF_COMPONENTS + [llm_status])


if __name__ == "__main__":
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ATTENZIONE: 'ffmpeg' / 'ffprobe' non trovati nel PATH. "
              "Installali (es. 'winget install Gyan.FFmpeg') o il montaggio fallira'.\n")
    demo.launch(inbrowser=True, css=CSS)
