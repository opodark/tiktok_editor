"""Pipeline di montaggio riutilizzabile.

Sia la CLI (`autoedit.cli`) sia la GUI (`gui.py`) costruiscono un
`RenderConfig` e chiamano `run_pipeline`, cosi' la logica di montaggio
vive in un solo posto.
"""
from __future__ import annotations

import logging
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .beats import detect_beats, get_cut_times
from .edl import build_segments
from .effects import EDIT_STYLES, finish_chain
from .media import load_media
from .overlays import prepare_watermark_png, render_title_png
from .render import Segment, concat_with_xfade, mux_audio, render_segments

log = logging.getLogger("autoedit")

# formato di output -> (larghezza, altezza) in pixel
ASPECTS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}

# callback di avanzamento: (frazione 0..1, messaggio) -> None
ProgressCB = Callable[[float, str], None]


@dataclass
class RenderConfig:
    media_dir: Path
    audio: Path
    out: Path = Path("output.mp4")
    order_file: Optional[Path] = None
    aspect: str = "9:16"
    fps: int = 30
    style: str = "vivid"
    cuts_per_beat: float = 1.0
    accent_every: int = 4
    xfade_duration: float = 0.18
    beat_offset: float = 0.0   # sposta la griglia di taglio di N secondi nel brano (+ = tagli piu' avanti)
    strong_beats_only: bool = False   # taglia solo sui beat con accento marcato
    dynamic_pacing: bool = False      # accelera nei tratti intensi, rallenta nei tratti calmi
    snap_to_onsets: bool = True       # aggancia i tagli sottodivisi ai transienti reali
    transition_mode: str = "auto"     # auto | cut | fade | slide | zoom | chaos
    shuffle: bool = False

    # --- stile di montaggio (ricetta) ---------------------------------
    # "custom" = usa i singoli parametri qui sotto; altrimenti una voce di
    # effects.EDIT_STYLES pilota movimento/transizioni/effetti/rifiniture.
    edit_style: str = "clean"

    # --- effetti (usati quando edit_style == "custom") --------------------
    motion: str = "kenburns"          # kenburns | zoom | sway | handheld | punch | bounce | static
    motion_intensity: float = 1.0     # 0.2 .. 2.5
    impact_effects: tuple = ()        # chiavi in effects.IMPACT_EFFECTS, usate sui beat forti
    variety: float = 0.3             # 0 = ogni clip uguale, 1 = grade/movimento cambiano spesso
    grain: float = 0.0               # 0 .. 1
    vignette: bool = False
    chromatic: bool = False           # leggera aberrazione cromatica su tutto
    slowmo: bool = False              # (custom) slow-mo + accelerazioni sulle clip video
    hold_prob: float = 0.0            # (custom) probabilita' clip "hero" tenuta 2 beat (0..0.4)
    max_duration: Optional[float] = None
    seed: Optional[int] = None
    keep_temp: bool = False
    jobs: int = 0
    chunk_size: int = 10

    # --- intro / audio -----------------------------------------------------
    audio_start: float = 0.0     # da che secondo del brano partire (salta l'intro)
    hook_hold: float = 0.0       # tieni fermo il primo clip N secondi prima che parta il montaggio

    # --- watermark / logo ------------------------------------------------
    watermark: Optional[Path] = None
    watermark_pos: str = "br"    # tl | tr | bl | br
    watermark_scale: float = 0.15
    watermark_opacity: float = 0.85

    # --- titolo in sovrimpressione -------------------------------------
    title_text: str = ""
    title_pos: str = "center"    # top | center | bottom
    title_start: float = 0.0
    title_duration: float = 2.5


_WM_POS = {"tl", "tr", "bl", "br"}


def _build_overlays(cfg: RenderConfig, target_w: int, target_h: int, work_dir: Path) -> list[dict]:
    """Prepara i PNG di watermark e titolo e ritorna le voci per mux_audio."""
    overlays: list[dict] = []

    if cfg.watermark and Path(cfg.watermark).is_file():
        wm_png = work_dir / "watermark.png"
        ww, wh = prepare_watermark_png(Path(cfg.watermark), target_w,
                                       max(0.03, min(0.6, cfg.watermark_scale)), wm_png)
        m = int(target_w * 0.04)
        pos = cfg.watermark_pos if cfg.watermark_pos in _WM_POS else "br"
        xy = {
            "tl": (m, m),
            "tr": (target_w - ww - m, m),
            "bl": (m, target_h - wh - m),
            "br": (target_w - ww - m, target_h - wh - m),
        }[pos]
        overlays.append({"png": wm_png, "x": xy[0], "y": xy[1],
                         "opacity": max(0.05, min(1.0, cfg.watermark_opacity))})

    if cfg.title_text and cfg.title_text.strip():
        ti_png = work_dir / "title.png"
        render_title_png(cfg.title_text, target_w, target_h, cfg.title_pos, ti_png)
        s = max(0.0, cfg.title_start)
        e = s + max(0.4, cfg.title_duration)
        overlays.append({"png": ti_png, "x": 0, "y": 0, "start": s, "end": e, "fade": 0.3})

    return overlays


def run_pipeline(cfg: RenderConfig, progress: Optional[ProgressCB] = None) -> Path:
    """Esegue l'intera pipeline e ritorna il percorso del video finale.

    `progress`, se passato, viene chiamato con (frazione, messaggio) a
    ogni tappa; gli stessi messaggi finiscono comunque nel logger.
    """
    def _p(frac: float, msg: str) -> None:
        log.info(msg)
        if progress is not None:
            progress(frac, msg)

    if cfg.aspect not in ASPECTS:
        raise ValueError(f"Formato non valido: {cfg.aspect}")

    rng = random.Random(cfg.seed)
    target_w, target_h = ASPECTS[cfg.aspect]

    _p(0.02, f"Carico i media da {cfg.media_dir} ...")
    media_items = load_media(cfg.media_dir, order_file=cfg.order_file)
    _p(0.08, "Trovati %d media (%d foto, %d video)." % (
        len(media_items),
        sum(1 for m in media_items if m.kind == "image"),
        sum(1 for m in media_items if m.kind == "video"),
    ))

    _p(0.10, f"Rilevo il beat di {cfg.audio} ...")
    beat_info = detect_beats(cfg.audio)
    _p(0.20, "Tempo stimato: %.1f BPM, %d beat (%d forti, %d downbeat), %d transienti, "
             "durata audio %.1fs" % (
        beat_info.tempo, len(beat_info.beat_times), len(beat_info.strong_times),
        len(beat_info.downbeat_times), len(beat_info.onset_times), beat_info.duration,
    ))

    cut_times = get_cut_times(
        beat_info, cfg.cuts_per_beat,
        strong_only=cfg.strong_beats_only,
        dynamic=cfg.dynamic_pacing,
        snap=cfg.snap_to_onsets,
    )
    if cfg.beat_offset:
        cut_times = cut_times - cfg.beat_offset
        cut_times = cut_times[cut_times >= 0.0]
    if cfg.audio_start and cfg.audio_start > 0:
        cut_times = cut_times[cut_times >= cfg.audio_start]
    if cfg.max_duration and len(cut_times):
        cut_times = cut_times[cut_times <= cut_times[0] + cfg.max_duration]
    if len(cut_times) < 2:
        raise ValueError("Troppo pochi punti di taglio: brano troppo corto, "
                         "audio-start / beat-offset / max-duration troppo aggressivi.")
    first_cut = float(cut_times[0])

    # ricetta ("clean"/"hype"/...) oppure None per la modalita' custom
    recipe = EDIT_STYLES.get(cfg.edit_style) if cfg.edit_style not in ("custom", "", None) else None
    if recipe:
        xfd = 0.0 if recipe["xfade_ms"] < 70 else recipe["xfade_ms"] / 1000.0
        finish = finish_chain(recipe["grain"], recipe["vignette"], recipe["chromatic"])
        motion_intensity = recipe["motion_intensity"]
        _p(0.21, f"Stile montaggio: {cfg.edit_style}")
    else:
        xfd = cfg.xfade_duration if cfg.transition_mode != "cut" else 0.0
        finish = finish_chain(grain=max(0.0, cfg.grain), vignette=bool(cfg.vignette),
                              chromatic=bool(cfg.chromatic))
        motion_intensity = cfg.motion_intensity

    segments = build_segments(
        media_items, cut_times, beat_info,
        accent_every=cfg.accent_every, shuffle=cfg.shuffle, rng=rng,
        transition_mode=cfg.transition_mode,
        base_style=cfg.style, base_motion=cfg.motion,
        variety=max(0.0, min(1.0, cfg.variety)),
        impact_pool=tuple(cfg.impact_effects or ()),
        recipe=recipe,
        video_speeds=([1.0, 1.0, 0.5, 2.0, 1.6] if cfg.slowmo else [1.0]),
        hold_prob=max(0.0, min(0.4, cfg.hold_prob)),
    )

    # "hook": un clip iniziale tenuto fermo prima che il montaggio parta.
    # L'audio viene fatto partire prima di altrettanto, cosi' il primo
    # taglio vero cade comunque sul beat.
    audio_offset = first_cut
    hold = max(0.0, cfg.hook_hold)
    if 0.0 < hold < 0.3:
        hold = 0.3
    if hold and segments:
        intro = Segment(item=segments[0].item, duration=hold,
                        transition="fade", accented=False, strong=False)
        segments.insert(0, intro)
        audio_offset = max(0.0, first_cut - hold)

    _p(0.25, "Costruiti %d segmenti (durata totale prevista %.1fs)." % (
        len(segments), sum(s.duration for s in segments),
    ))

    work_dir = Path(tempfile.mkdtemp(prefix="autoedit_"))
    try:
        _p(0.30, f"Rendering dei segmenti in {work_dir} ...")
        render_segments(segments, target_w, target_h, cfg.fps, cfg.style, rng, work_dir,
                        jobs=cfg.jobs, xfade_duration=xfd, chunk_size=cfg.chunk_size,
                        motion=cfg.motion, motion_intensity=motion_intensity, finish=finish)

        silent_path = work_dir / "silent.mp4"
        _p(0.80, "Assemblo i segmenti con le transizioni ...")
        total_dur = concat_with_xfade(segments, xfd, silent_path, cfg.fps,
                                      work_dir=work_dir, chunk_size=cfg.chunk_size,
                                      target_w=target_w, target_h=target_h)
        _p(0.92, "Montaggio video pronto (%.2fs)." % total_dur)

        overlays = _build_overlays(cfg, target_w, target_h, work_dir)

        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        _p(0.95, "Audio + testo/logo ..." if overlays else "Aggiungo la traccia audio ...")
        mux_audio(silent_path, cfg.audio, cfg.out, total_dur, fade_in=0.15, fade_out=1.5,
                  audio_offset=audio_offset, overlays=overlays)
        _p(1.0, f"Fatto -> {cfg.out}")
    finally:
        if not cfg.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            log.info("File temporanei conservati in %s", work_dir)

    return cfg.out
