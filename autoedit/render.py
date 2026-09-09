"""Rendering dei singoli segmenti (foto/video) e assemblaggio finale.

Sincronizzazione col beat
-------------------------
Ogni transizione xfade fa "sovrapporre" la coda di un segmento con la
testa del successivo: in quella finestra il video avanza ma la timeline
di uscita si accorcia di `xf` secondi. Se non si compensa, l'errore si
somma taglio dopo taglio e il montaggio finisce diversi secondi in
anticipo sulla musica.

Compensazione: ogni segmento viene renderizzato con una CODA extra lunga
quanto la transizione che lo segue (`render_duration = duration + pad`).
La transizione consuma quella coda invece del contenuto "buono", quindi
il taglio visibile cade esattamente sul beat e la durata totale del
video coincide con l'intervallo di beat coperto.
"""
from __future__ import annotations

import logging
import os
import random
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .effects import (
    COLOR_STYLES, IMPACT_EFFECTS, crop_to_fill, kenburns_filter, video_motion_filter,
)
from .media import MediaItem

log = logging.getLogger("autoedit")

# Sotto questa soglia la transizione e' considerata uno "stacco netto":
# xfade con durata < ~2 frame si comporta male in ffmpeg, quindi in quel
# caso i segmenti vengono concatenati senza sovrapposizione (hard cut).
_XF_FLOOR = 0.07
# durata minima usata per una vera dissolvenza (>= 2 frame a 30 fps).
_MIN_XF = _XF_FLOOR


def _is_hard(xfade_duration: float) -> bool:
    return xfade_duration < _XF_FLOOR


@dataclass
class Segment:
    item: MediaItem
    duration: float                       # durata "utile" del segmento: la distanza fra due beat
    transition: str                       # transizione xfade per ENTRARE in questo segmento
    accented: bool
    strong: bool = False                  # taglio sul downbeat: "punch" di zoom piu' marcato
    color_style: Optional[str] = None     # grade specifico del segmento (None = usa quello base)
    motion: Optional[str] = None          # movimento specifico del segmento (None = usa quello base)
    impact_key: str = ""                  # effetto d'impatto sui primi frame (solo tagli accentati)
    video_speed: float = 1.0              # <1 = slow motion, >1 = accelerato (solo clip video)
    hero: bool = False                    # segmento "tenuto" piu' a lungo (2 beat)
    clip_path: Optional[Path] = None
    render_duration: Optional[float] = None  # durata realmente renderizzata (duration + coda per la transizione)

    @property
    def rdur(self) -> float:
        return self.render_duration if self.render_duration is not None else self.duration


def _run(cmd: List[str]) -> None:
    log.debug("ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def _boundary_xfade(prev_core: float, next_core: float, xfade_duration: float) -> float:
    """Durata della transizione fra due segmenti: mai piu' del 40% del
    segmento piu' corto, cosi' non "mangia" contenuto utile."""
    return max(_MIN_XF, min(max(xfade_duration, _MIN_XF), prev_core * 0.4, next_core * 0.4))


def build_image_segment(item: MediaItem, duration: float, target_w: int, target_h: int,
                         fps: int, style: str, rng: random.Random, punch_in: bool,
                         out_path: Path, pad: float = 0.0, strong: bool = False,
                         motion: str = "kenburns", motion_intensity: float = 1.0,
                         impact: str = "", finish: str = "") -> float:
    """Renderizza una foto come clip di `duration + pad` secondi.
    Ritorna la durata effettivamente renderizzata."""
    total = duration + pad
    fill = crop_to_fill(item.width, item.height, target_w, target_h)
    kb = kenburns_filter(target_w, target_h, total, fps, rng, punch_in=punch_in, strong=strong,
                         motion=motion, intensity=motion_intensity)
    color = COLOR_STYLES.get(style, "")
    parts = [fill, kb]
    if color:
        parts.append(color)
    if finish:
        parts.append(finish)
    if impact:
        parts.append(impact)
    parts.append("format=yuv420p")
    vf = ",".join(parts)
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(item.path),
        "-vf", vf, "-t", f"{total:.3f}", "-an", str(out_path), "-loglevel", "error",
    ]
    _run(cmd)
    return total


def build_video_segment(item: MediaItem, duration: float, target_w: int, target_h: int,
                         fps: int, style: str, rng: random.Random,
                         out_path: Path, pad: float = 0.0,
                         impact: str = "", finish: str = "", speed: float = 1.0,
                         video_motion: bool = True, motion_intensity: float = 1.0,
                         punch: bool = False, strong: bool = False,
                         best_start: Optional[float] = None) -> tuple[float, float]:
    """Renderizza uno spezzone video di `duration + pad` secondi.

    `speed` < 1 = slow motion (serve meno girato), > 1 = accelerato.
    `best_start` (istante "interessante" della clip, stimato al caricamento)
    viene usato come punto d'attacco invece di uno a caso.
    Se manca girato, l'ultimo frame viene congelato (tpad) per non far
    slittare i tagli successivi.
    Ritorna (durata_utile, durata_renderizzata)."""
    avail = item.duration or duration
    core = duration
    want = core + pad
    speed = max(0.25, min(3.0, float(speed)))
    src_needed = want * speed                       # secondi di girato richiesti
    if src_needed > avail and speed < 1.0:          # non basta per lo slow-mo -> avvicina a 1x
        speed = min(1.0, avail / max(want, 1e-3))
        src_needed = want * speed

    max_start = max(0.0, avail - src_needed)
    if best_start is not None and max_start > 0:
        jitter = rng.uniform(-0.15, 0.15) * min(1.0, max_start)
        start = min(max_start, max(0.0, float(best_start) + jitter))
    else:
        start = rng.uniform(0, max_start) if max_start > 0 else 0.0

    real_src = max(0.0, min(src_needed, avail - start))
    shortfall = max(0.0, want - real_src / speed)

    parts = [crop_to_fill(item.width, item.height, target_w, target_h)]
    color = COLOR_STYLES.get(style, "")
    if color:
        parts.append(color)
    if finish:
        parts.append(finish)
    if video_motion:
        parts.append(video_motion_filter(target_w, target_h, fps, rng,
                                         punch=punch, strong=strong, intensity=motion_intensity))
    # lo stretch temporale va DOPO zoompan (che altrimenti ricostruisce i
    # timestamp dal conteggio frame e annullerebbe il setpts), poi si
    # ricampiona a fps costante.
    if abs(speed - 1.0) > 1e-3:
        parts.append(f"setpts=PTS/{speed:.4f}")      # speed=0.5 -> PTS*2 (piu' lento)
        parts.append(f"fps={fps}")
    elif not video_motion:
        parts.append(f"fps={fps}")
    if impact:
        parts.append(impact)
    if shortfall > 1e-3:
        parts.append(f"tpad=stop_mode=clone:stop_duration={shortfall:.3f}")
    parts.append("format=yuv420p")
    vf = ",".join(parts)

    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{src_needed:.3f}", "-i", str(item.path),
        "-vf", vf, "-t", f"{want:.3f}", "-an", str(out_path), "-loglevel", "error",
    ]
    _run(cmd)
    return core, want


def _render_one(i: int, seg: Segment, target_w: int, target_h: int, fps: int, style: str,
                 seed: int, work_dir: Path, pad: float,
                 motion: str, motion_intensity: float, finish: str) -> None:
    # ogni worker ha il proprio Random derivato dal seed globale + indice,
    # cosi' il risultato resta riproducibile anche in parallelo.
    local_rng = random.Random(seed + i if seed is not None else None)
    out_path = work_dir / f"seg{i:03d}.mp4"
    eff_style = seg.color_style or style
    eff_motion = seg.motion or motion
    impact = IMPACT_EFFECTS.get(seg.impact_key, "")
    if seg.item.kind == "image":
        rdur = build_image_segment(seg.item, seg.duration, target_w, target_h, fps, eff_style,
                                    local_rng, punch_in=seg.accented, out_path=out_path, pad=pad,
                                    strong=seg.strong, motion=eff_motion,
                                    motion_intensity=motion_intensity, impact=impact, finish=finish)
        seg.render_duration = rdur
    else:
        core, rdur = build_video_segment(
            seg.item, seg.duration, target_w, target_h, fps, eff_style, local_rng,
            out_path=out_path, pad=pad, impact=impact, finish=finish,
            speed=seg.video_speed, motion_intensity=motion_intensity,
            punch=seg.accented, strong=seg.strong,
            best_start=getattr(seg.item, "best_start", None),
        )
        seg.duration = core
        seg.render_duration = rdur
    seg.clip_path = out_path


def _plan_pads(segments: List[Segment], xfade_duration: float, chunk_size: int) -> List[float]:
    """Coda extra da renderizzare per ogni segmento = durata della
    transizione che lo segue. Ai confini di chunk si usa la transizione
    piena, perche' li' i segmenti confinanti sono lunghi (chunk interi)."""
    if _is_hard(xfade_duration):
        return [0.0] * len(segments)        # stacchi netti: nessuna coda da renderizzare

    pads: List[float] = []
    n = len(segments)
    for i in range(n):
        if i == n - 1:
            pads.append(0.0)                       # ultimo segmento: nessuna transizione dopo
        elif chunk_size and (i + 1) % chunk_size == 0:
            pads.append(max(xfade_duration, _MIN_XF))   # confine di chunk -> transizione piena
        else:
            pads.append(_boundary_xfade(segments[i].duration, segments[i + 1].duration,
                                        xfade_duration))
    return pads


def render_segments(segments: List[Segment], target_w: int, target_h: int, fps: int,
                     style: str, rng: random.Random, work_dir: Path,
                     jobs: int = 0, xfade_duration: float = 0.18, chunk_size: int = 10,
                     motion: str = "kenburns", motion_intensity: float = 1.0,
                     finish: str = "") -> None:
    """Renderizza tutti i segmenti, in parallelo su piu' processi ffmpeg."""
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs or min(8, (os.cpu_count() or 4))
    seed = rng.randrange(1 << 30)
    pads = _plan_pads(segments, xfade_duration, chunk_size)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {
            ex.submit(_render_one, i, seg, target_w, target_h, fps, style, seed, work_dir,
                      pads[i], motion, motion_intensity, finish): i
            for i, seg in enumerate(segments)
        }
        done = 0
        for fut in as_completed(futures):
            fut.result()  # propaga eventuali eccezioni
            done += 1
            log.info("segmento renderizzato (%d/%d)", done, len(segments))

    for i, seg in enumerate(segments):
        log.debug("seg %03d: %s %.2fs mov=%s tr=%s%s%s%s%s", i, seg.item.kind, seg.duration,
                   seg.motion or "-", seg.transition,
                   f" x{seg.video_speed:g}" if seg.item.kind == "video" and seg.video_speed != 1.0 else "",
                   " HERO" if seg.hero else "",
                   f" IMPATTO={seg.impact_key}" if seg.impact_key else "",
                   " [DOWNBEAT]" if seg.strong else (" [acc]" if seg.accented else ""))


def _hard_concat(clips: List[Segment], out_path: Path, fps: int,
                  target_w: Optional[int] = None, target_h: Optional[int] = None) -> float:
    """Concatena i clip senza sovrapposizione (stacco netto). Il filtro
    `concat` e' pignolo su dimensioni/SAR/fps, quindi normalizziamo ogni
    input prima di unirlo. Durata di uscita = somma delle durate utili."""
    if len(clips) == 1:
        _run(["ffmpeg", "-y", "-i", str(clips[0].clip_path), "-c", "copy",
              str(out_path), "-loglevel", "error"])
        return clips[0].rdur

    inputs: List[str] = []
    for c in clips:
        inputs += ["-i", str(c.clip_path)]

    norm = f"fps={fps},setsar=1,format=yuv420p"
    if target_w and target_h:
        norm = f"scale={target_w}:{target_h}," + norm
    pre = ";".join(f"[{i}:v]{norm}[c{i}]" for i in range(len(clips)))
    labels = "".join(f"[c{i}]" for i in range(len(clips)))
    fc = f"{pre};{labels}concat=n={len(clips)}:v=1:a=0[vout]"
    _run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[vout]",
          "-r", str(fps), "-pix_fmt", "yuv420p", str(out_path), "-loglevel", "error"])
    return sum(c.rdur for c in clips)


def _xfade_chain(clips: List[Segment], xfade_duration: float, out_path: Path, fps: int) -> float:
    """Concatena una lista di clip (segmenti o chunk gia' assemblati) con
    transizioni xfade in un unico filter_complex. Va tenuta corta (poche
    decine di input al massimo): ffmpeg deve tenere in memoria tutti gli
    stream aperti contemporaneamente, e con troppi input il processo
    puo' esaurire la RAM ed essere ucciso dal kernel (SIGKILL).

    Ritorna la durata di uscita, che -- grazie alla coda extra su ogni
    clip -- coincide con la somma delle durate "utili" (cioe' con
    l'intervallo di beat coperto)."""
    if len(clips) == 1:
        cmd = ["ffmpeg", "-y", "-i", str(clips[0].clip_path), "-c", "copy",
               str(out_path), "-loglevel", "error"]
        _run(cmd)
        return clips[0].rdur

    inputs: List[str] = []
    for c in clips:
        inputs += ["-i", str(c.clip_path)]

    filters = []
    run = clips[0].rdur                      # lunghezza corrente dello stream di uscita
    prev = "0:v"
    for k in range(1, len(clips)):
        prev_core = clips[k - 1].duration
        tail_pad = max(0.0, clips[k - 1].rdur - prev_core)
        xf = _boundary_xfade(prev_core, clips[k].duration, xfade_duration)
        if tail_pad > 0:
            xf = min(xf, tail_pad)          # non superare la coda realmente renderizzata
        xf = max(xf, 0.01)
        offset = max(0.0, run - xf)         # la transizione parte sul beat (fine della durata utile)
        out_label = f"v{k}" if k < len(clips) - 1 else "vout"
        tr = clips[k].transition
        filters.append(
            f"[{prev}][{k}:v]xfade=transition={tr}:duration={xf:.3f}:offset={offset:.3f}[{out_label}]"
        )
        run = run - xf + clips[k].rdur
        prev = out_label

    filter_complex = ";".join(filters)
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex, "-map", "[vout]",
        "-r", str(fps), "-pix_fmt", "yuv420p", str(out_path), "-loglevel", "error",
    ]
    _run(cmd)
    return run


def concat_with_xfade(segments: List[Segment], xfade_duration: float, out_path: Path,
                       fps: int, work_dir: Optional[Path] = None, chunk_size: int = 10,
                       target_w: Optional[int] = None, target_h: Optional[int] = None) -> float:
    """Assembla tutti i segmenti con transizioni xfade, a prescindere da
    quanti siano: li raggruppa in chunk piccoli (<= chunk_size), assembla
    ogni chunk in un file, poi fonde ricorsivamente i chunk fra loro.
    Cosi' nessuna singola chiamata ffmpeg tiene aperti troppi stream."""
    hard = _is_hard(xfade_duration)

    if len(segments) <= chunk_size:
        return (_hard_concat(segments, out_path, fps, target_w, target_h) if hard
                else _xfade_chain(segments, xfade_duration, out_path, fps))

    assert work_dir is not None, "work_dir richiesto quando i segmenti superano chunk_size"

    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
    pseudo: List[Segment] = []
    for ci, chunk in enumerate(chunks):
        chunk_out = work_dir / f"chunk_{uuid.uuid4().hex[:8]}.mp4"
        if hard:
            rendered = _hard_concat(chunk, chunk_out, fps, target_w, target_h)
            core = rendered                     # nessuna coda: durata gia' esatta
        else:
            rendered = _xfade_chain(chunk, xfade_duration, chunk_out, fps)
            is_last_chunk = ci == len(chunks) - 1
            # la coda extra del chunk (tranne l'ultimo) e' lunga una transizione
            # piena: verra' consumata fondendo questo chunk col successivo.
            core = rendered if is_last_chunk else max(0.0, rendered - max(xfade_duration, _MIN_XF))
        pseudo.append(Segment(item=chunk[0].item, duration=core, render_duration=rendered,
                               transition=chunk[0].transition, accented=False,
                               clip_path=chunk_out))

    return concat_with_xfade(pseudo, xfade_duration, out_path, fps, work_dir, chunk_size,
                             target_w, target_h)


def mux_audio(video_path: Path, audio_path: Path, out_path: Path, total_duration: float,
              fade_in: float, fade_out: float, audio_offset: float = 0.0,
              overlays: Optional[List[dict]] = None) -> None:
    """Passata finale: attacca l'audio al video muto e, in un colpo solo,
    sovrappone eventuali PNG (titolo, watermark) cosi' si ricodifica una
    volta sola.

    `audio_offset` e' l'istante del brano che deve coincidere col primo
    frame del video (il primo beat usato per i tagli), cosi' il "uno"
    della musica cade sul primo taglio invece che 0.2-0.5s prima.

    Ogni voce di `overlays` e' un dict:
        png      percorso del PNG (con alpha)
        x, y     posizione in pixel
        opacity  0..1 (default 1)
        start,end secondi: se presenti, l'overlay compare solo in [start,end]
        fade     secondi di dissolvenza in/out dell'alpha (default 0)
    """
    overlays = overlays or []
    start = max(0.0, audio_offset)
    end = start + total_duration
    fade_out_start = max(0.0, total_duration - fade_out)
    af = (
        f"atrim=start={start:.3f}:end={end:.3f},asetpts=N/SR/TB,"
        f"afade=t=in:st=0:d={fade_in:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}"
    )

    # ogni PNG entra come stream in loop: cosi' i filtri temporali
    # (fade dell'alpha, enable) hanno una timeline continua su cui
    # lavorare invece di un singolo fotogramma.
    inputs = ["-i", str(video_path), "-i", str(audio_path)]
    for ov in overlays:
        inputs += ["-loop", "1", "-i", str(ov["png"])]

    fc = [f"[1:a]{af}[a]"]
    vlab = "0:v"
    for i, ov in enumerate(overlays):
        idx = 2 + i
        chain = f"[{idx}:v]format=rgba"
        if float(ov.get("opacity", 1.0)) < 1.0:
            chain += f",colorchannelmixer=aa={float(ov['opacity']):.3f}"
        s, e, fd = ov.get("start"), ov.get("end"), float(ov.get("fade", 0.0))
        if fd > 0 and s is not None and e is not None:
            chain += (f",fade=t=in:st={float(s):.3f}:d={fd:.3f}:alpha=1"
                      f",fade=t=out:st={max(float(s), float(e) - fd):.3f}:d={fd:.3f}:alpha=1")
        chain += f"[ov{i}]"
        fc.append(chain)
        enable = ""
        if s is not None and e is not None:
            enable = f":enable='between(t,{float(s):.3f},{float(e):.3f})'"
        out_lab = f"vv{i}"
        fc.append(f"[{vlab}][ov{i}]overlay={int(ov['x'])}:{int(ov['y'])}{enable}[{out_lab}]")
        vlab = out_lab

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(fc),
        "-map", f"[{vlab}]" if overlays else "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out_path), "-loglevel", "error",
    ]
    _run(cmd)
