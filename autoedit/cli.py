"""Entry point da riga di comando."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .effects import COLOR_STYLES, EDIT_STYLES, IMPACT_EFFECTS, MOTION_MODES
from .pipeline import ASPECTS, RenderConfig, run_pipeline

log = logging.getLogger("autoedit")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoedit",
        description="Montaggio automatico stile TikTok/CapCut: foto+video, tagli sincronizzati "
                    "sul beat reale di un brano, Ken Burns randomizzato, color grading, "
                    "transizioni glitch/flash sugli accenti.",
    )
    p.add_argument("--media-dir", required=True, type=Path, help="Cartella con foto/video da montare")
    p.add_argument("--audio", required=True, type=Path, help="File audio (mp3/wav) da cui rilevare il beat")
    p.add_argument("--out", type=Path, default=Path("output.mp4"), help="File video di output")
    p.add_argument("--order-file", type=Path, default=None,
                    help="File di testo opzionale con l'ordine dei nomi file, uno per riga")

    p.add_argument("--aspect", choices=list(ASPECTS), default="9:16", help="Formato di output")
    p.add_argument("--fps", type=int, default=30, help="Frame rate di output")

    p.add_argument("--crop-zoom", type=float, default=1.0,
                    help="Inquadratura: >= 1 stringe il ritaglio (1.0 = riempi e basta).")
    p.add_argument("--crop-x", type=float, default=0.5,
                    help="Ritaglio orizzontale 0..1 (0 = sinistra, 0.5 = centro, 1 = destra).")
    p.add_argument("--crop-y", type=float, default=0.5,
                    help="Ritaglio verticale 0..1 (0 = alto, 0.5 = centro, 1 = basso).")

    p.add_argument("--style", choices=list(COLOR_STYLES), default="vivid",
                    help="Color grading di base")
    p.add_argument("--cuts-per-beat", type=float, default=1.0,
                    help="1 = un taglio a beat, 0.5 = un taglio ogni 2 beat, 2 = un taglio ogni ottavo")
    p.add_argument("--accent-every", type=int, default=4,
                    help="Ogni quanti tagli forzare un accento (flash/glitch + zoom punch), oltre a "
                         "quelli rilevati automaticamente dal beat forte. 0 = disattiva il forzato")
    p.add_argument("--xfade-duration", type=float, default=0.18, help="Durata delle transizioni (s)")
    p.add_argument("--beat-offset", type=float, default=0.0,
                    help="Micro-regolazione della sincronia in secondi: sposta la griglia dei "
                         "tagli avanti (valori positivi) o indietro (negativi) nel brano. "
                         "Utile se i tagli 'sentono' leggermente in anticipo/ritardo.")
    p.add_argument("--strong-beats-only", action="store_true",
                    help="Taglia solo sui beat con accento marcato (montaggio piu' calmo).")
    p.add_argument("--dynamic-pacing", action="store_true",
                    help="Accelera i tagli nei tratti a energia alta del brano e li rallenta "
                         "nei tratti calmi, attorno al valore di --cuts-per-beat.")
    p.add_argument("--no-snap", dest="snap", action="store_false",
                    help="Non agganciare i tagli sottodivisi ai transienti reali del brano.")
    p.set_defaults(snap=True)
    p.add_argument("--transitions", choices=["auto", "cut", "fade", "slide", "zoom", "chaos"],
                    default="auto", help="Stile delle transizioni fra i tagli.")

    p.add_argument("--edit-style", choices=["custom", *EDIT_STYLES], default="clean",
                    help="Stile di montaggio (ricetta coerente stile TikTok/CapCut). "
                         "'custom' = usa i singoli parametri qui sotto.")
    p.add_argument("--motion", choices=MOTION_MODES, default="kenburns",
                    help="Tipo di movimento sulle clip (solo con --edit-style custom).")
    p.add_argument("--motion-intensity", type=float, default=1.0,
                    help="Intensita' del movimento 0.2..2.5 (default 1.0).")
    p.add_argument("--impact", nargs="*", choices=list(IMPACT_EFFECTS), default=[],
                    metavar="EFFETTO",
                    help="Effetti d'impatto usati a caso sui tagli accentati: "
                         + " ".join(IMPACT_EFFECTS) + ". Nessuno = disattiva.")
    p.add_argument("--variety", type=float, default=0.3,
                    help="0 = ogni clip con lo stesso grade/movimento, 1 = cambiano spesso.")
    p.add_argument("--grain", type=float, default=0.0, help="Grana 0..1.")
    p.add_argument("--vignette", action="store_true", help="Aggiunge una vignettatura.")
    p.add_argument("--chromatic", action="store_true",
                    help="Leggera aberrazione cromatica su tutto il video.")
    p.add_argument("--slowmo", action="store_true",
                    help="(edit-style custom) slow-motion e accelerazioni sulle clip video.")
    p.add_argument("--hold-prob", type=float, default=0.0,
                    help="(edit-style custom) probabilita' 0..0.4 che una clip sia tenuta 2 beat.")

    p.add_argument("--audio-start", type=float, default=0.0,
                    help="Da che secondo del brano partire (salta l'intro).")
    p.add_argument("--hook-hold", type=float, default=0.0,
                    help="Tieni fermo il primo clip N secondi prima che parta il montaggio "
                         "(hook). L'audio parte prima di altrettanto, il primo taglio resta sul beat.")

    p.add_argument("--watermark", type=Path, default=None, help="PNG (con trasparenza) da usare come logo.")
    p.add_argument("--watermark-pos", choices=["tl", "tr", "bl", "br"], default="br",
                    help="Angolo del watermark (default: basso-destra).")
    p.add_argument("--watermark-scale", type=float, default=0.15,
                    help="Larghezza del watermark come frazione del video (default 0.15).")
    p.add_argument("--watermark-opacity", type=float, default=0.85, help="Opacita' del watermark 0..1.")

    p.add_argument("--title", dest="title_text", default="", help="Testo del titolo in sovrimpressione.")
    p.add_argument("--title-pos", choices=["top", "center", "bottom"], default="center")
    p.add_argument("--title-start", type=float, default=0.0, help="Quando compare il titolo (s).")
    p.add_argument("--title-duration", type=float, default=2.5, help="Per quanto resta il titolo (s).")

    p.add_argument("--shuffle", action="store_true", help="Mischia l'ordine dei media")
    p.add_argument("--max-duration", type=float, default=None,
                    help="Taglia il montaggio a al massimo N secondi")
    p.add_argument("--seed", type=int, default=None, help="Seed random per risultati riproducibili")
    p.add_argument("--keep-temp", action="store_true", help="Non cancellare i file temporanei")
    p.add_argument("--jobs", type=int, default=0,
                    help="Quanti segmenti renderizzare in parallelo (0 = auto)")
    p.add_argument("--chunk-size", type=int, default=10,
                    help="Max segmenti per chiamata ffmpeg nell'assemblaggio a catena "
                         "(piu' alto = meno file intermedi ma piu' RAM richiesta)")
    p.add_argument("-v", "--verbose", action="store_true")

    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    # con -v il nostro logger va in DEBUG, ma non vogliamo il diluvio di
    # bytecode di numba/librosa quando compilano i kernel di beat detection.
    for noisy in ("numba", "librosa", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = RenderConfig(
        media_dir=args.media_dir,
        audio=args.audio,
        out=args.out,
        order_file=args.order_file,
        aspect=args.aspect,
        fps=args.fps,
        crop_zoom=args.crop_zoom,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        style=args.style,
        cuts_per_beat=args.cuts_per_beat,
        accent_every=args.accent_every,
        xfade_duration=args.xfade_duration,
        beat_offset=args.beat_offset,
        strong_beats_only=args.strong_beats_only,
        dynamic_pacing=args.dynamic_pacing,
        snap_to_onsets=args.snap,
        transition_mode=args.transitions,
        edit_style=args.edit_style,
        motion=args.motion,
        motion_intensity=args.motion_intensity,
        impact_effects=tuple(args.impact),
        variety=args.variety,
        grain=args.grain,
        vignette=args.vignette,
        chromatic=args.chromatic,
        slowmo=args.slowmo,
        hold_prob=args.hold_prob,
        audio_start=args.audio_start,
        hook_hold=args.hook_hold,
        watermark=args.watermark,
        watermark_pos=args.watermark_pos,
        watermark_scale=args.watermark_scale,
        watermark_opacity=args.watermark_opacity,
        title_text=args.title_text,
        title_pos=args.title_pos,
        title_start=args.title_start,
        title_duration=args.title_duration,
        shuffle=args.shuffle,
        max_duration=args.max_duration,
        seed=args.seed,
        keep_temp=args.keep_temp,
        jobs=args.jobs,
        chunk_size=args.chunk_size,
    )

    try:
        run_pipeline(cfg)
    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
