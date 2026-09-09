"""Costruisce la Edit Decision List: quale media va in quale slot, per
quanto, con quale movimento / transizione / effetto.

Puo' lavorare in due modi:
- "custom": ogni parametro (stile colore, movimento, transizioni, pool di
  effetti) arriva dai controlli e viene randomizzato in base a `variety`;
- con una "ricetta" (EDIT_STYLES): il comportamento e' coerente e vicino
  ai preset TikTok/CapCut -- per lo piu' stacchi netti sul beat, movimento
  che pesca da un pool ristretto (mai lo stesso due volte di fila), ed
  effetti d'impatto solo sui colpi forti, distanziati nel tempo.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np

from .beats import BeatInfo, _energy_at, is_accented, is_downbeat
from .effects import pick_color_style, pick_no_repeat, pick_transition
from .media import MediaItem
from .render import Segment

# transizioni "morbide" per la ricetta smooth / modalita' custom non-hard
_SOFT_XFADES = ["fade", "dissolve", "smoothleft", "smoothright", "circleopen", "slideup"]
_CALM_MOTIONS = ("kenburns", "sway", "static")
_HYPE_MOTIONS = ("punch", "bounce", "handheld")


def build_segments(
    media_items: List[MediaItem],
    cut_times: np.ndarray,
    beat_info: BeatInfo,
    accent_every: int,
    shuffle: bool,
    rng: random.Random,
    min_segment_dur: float = 0.18,
    transition_mode: str = "auto",
    base_style: str = "vivid",
    base_motion: str = "kenburns",
    variety: float = 0.0,
    impact_pool: Sequence[str] = (),
    recipe: Optional[dict] = None,
    video_speeds: Optional[Sequence[float]] = None,   # solo modalita' custom
    hold_prob: float = 0.0,                            # solo modalita' custom
) -> List[Segment]:
    if len(cut_times) < 2:
        raise ValueError("Servono almeno due punti di taglio (beat) per montare qualcosa.")

    items = list(media_items)
    if shuffle:
        rng.shuffle(items)
    if not items:
        raise ValueError("Nessun media da montare.")

    # --- parametri effettivi: dalla ricetta se c'e', altrimenti dai controlli ---
    if recipe:
        motions = list(recipe["motions"])
        impacts = list(recipe["impacts"])
        impact_on = recipe["impact_on"]
        impact_min_gap = float(recipe["impact_min_gap"])
        eff_variety = float(recipe["variety"])
        hard_cuts = recipe["xfade_ms"] < 70
        xfade_pool = list(recipe["xfade_pool"]) if recipe.get("xfade_pool") else _SOFT_XFADES
        video_speeds = list(recipe.get("video_speeds") or [1.0])
        hold_prob = float(recipe.get("hold_prob") or 0.0)
    else:
        motions = [base_motion]
        impacts = [e for e in impact_pool if e]
        impact_on = "every_strong" if impacts else "off"
        impact_min_gap = 0.0
        eff_variety = float(variety)
        hard_cuts = transition_mode == "cut"
        xfade_pool = None  # -> pick_transition col suo mode
        video_speeds = list(video_speeds) if video_speeds else [1.0]
        hold_prob = float(hold_prob or 0.0)

    # energia per taglio: per capire i tratti "calmi" vs "carichi"
    cut_energy = np.array([_energy_at(beat_info, float(t)) for t in cut_times])
    e_hi = float(np.percentile(cut_energy, 68)) if len(cut_energy) else 0.0
    e_lo = float(np.percentile(cut_energy, 33)) if len(cut_energy) else 0.0

    segments: List[Segment] = []
    media_i = 0
    prev_transition = prev_motion = prev_impact = None
    last_impact_at = -1e9
    clock = 0.0

    k = 1
    n = len(cut_times)
    while k < n:
        dur = float(cut_times[k] - cut_times[k - 1])
        if dur < min_segment_dur:
            k += 1
            continue

        item = items[media_i % len(items)]
        media_i += 1

        # l'energia / il downbeat contano rispetto all'INIZIO del segmento (cut k-1),
        # perche' e' li' che scattano impatto e "punch".
        t_start = float(cut_times[k - 1])
        energy = float(cut_energy[k - 1])
        calm = energy <= e_lo
        hype = energy >= e_hi

        # --- clip "hero": la teniamo 2 beat (preferibilmente video o tratti calmi) ---
        hero = False
        if (hold_prob and k + 1 < n and rng.random() < hold_prob
                and (item.kind == "video" or calm)):
            nxt = float(cut_times[k + 1] - cut_times[k])
            if nxt >= min_segment_dur:
                dur += nxt
                hero = True
        clock += dur

        strong = is_downbeat(t_start, beat_info)
        hit = strong and is_accented(t_start, beat_info)        # colpo "vero"
        forced = accent_every > 0 and k % accent_every == 0 and not recipe
        accented = strong or hit or forced

        # --- transizione ---
        if hard_cuts:
            transition = "fade"                                  # durata forzata a ~0 altrove
        elif xfade_pool:
            transition = pick_no_repeat(xfade_pool, rng, prev_transition)
            prev_transition = transition
        else:
            transition = pick_transition(rng, accented, mode=transition_mode)
            prev_transition = transition

        # --- movimento (mai lo stesso due volte di fila) ---
        if eff_variety <= 0.05 and not recipe:
            motion = base_motion
        else:
            if calm:
                pref = [m for m in motions if m in _CALM_MOTIONS] or motions
            elif hype:
                pref = [m for m in motions if m in _HYPE_MOTIONS] or motions
            else:
                pref = motions
            motion = pick_no_repeat(pref, rng, prev_motion) or base_motion
        prev_motion = motion

        # --- effetto d'impatto: solo sui momenti giusti, e distanziati ---
        impact_key = ""
        if impacts and not calm:
            want = (
                (impact_on == "hits" and hit)
                or (impact_on == "downbeats" and strong)
                or (impact_on == "every_strong" and accented)
            )
            if want and (clock - last_impact_at) >= impact_min_gap:
                impact_key = pick_no_repeat(impacts, rng, prev_impact) or ""
                if impact_key:
                    prev_impact = impact_key
                    last_impact_at = clock

        # --- velocita' clip video (slow-mo / accelerato) ---
        vspeed = 1.0
        if item.kind == "video":
            if calm:                                             # nei tratti calmi niente accelerazioni
                vspeed = rng.choice([s for s in video_speeds if s <= 1.0] or [1.0])
            else:
                vspeed = rng.choice(video_speeds)

        color_style = pick_color_style(rng, base_style, eff_variety)

        # inquadratura leggermente diversa a ogni taglio: rende meno "uguali"
        # le clip, soprattutto con foto in raffica (fotogrammi quasi identici).
        jv = eff_variety
        jitter = (round(rng.uniform(0.0, 0.16) * jv, 3),
                  round(rng.uniform(-0.12, 0.12) * jv, 3),
                  round(rng.uniform(-0.12, 0.12) * jv, 3))

        segments.append(Segment(
            item=item, duration=dur, transition=transition, accented=accented, strong=strong,
            color_style=color_style, motion=motion, impact_key=impact_key,
            video_speed=vspeed, hero=hero, crop_jitter=jitter,
        ))
        k += 2 if hero else 1

    return segments
