"""Filtri ffmpeg: color grading, movimenti, effetti d'impatto, transizioni."""
from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Color grading. Ogni voce e' un frammento di catena -vf (eq/curves/...).
# ---------------------------------------------------------------------------
COLOR_STYLES = {
    "none": "",
    "vivid": "eq=contrast=1.12:saturation=1.35:brightness=0.01,curves=preset=increase_contrast",
    "cinematic": "eq=contrast=1.15:saturation=0.92:gamma=0.95,"
                 "curves=r='0/0 0.5/0.46 1/0.98':b='0/0.02 0.5/0.5 1/0.95'",
    "vhs": "eq=contrast=1.05:saturation=1.15:gamma=1.05,noise=alls=8:allf=t,curves=preset=vintage",
    "warm": "eq=contrast=1.08:saturation=1.20:gamma_r=1.06:gamma_b=0.94,"
            "colorbalance=rs=.06:bs=-.06:rm=.05:bm=-.05:rh=.03:bh=-.03",
    "cold": "eq=contrast=1.12:saturation=1.05,"
            "colorbalance=rs=-.06:bs=.08:rm=-.04:bm=.06:rh=-.03:bh=.05",
    "film": "curves=preset=lighter,eq=contrast=1.04:saturation=0.88:gamma=1.03,noise=alls=6:allf=t",
    "bw": "hue=s=0,eq=contrast=1.28:brightness=0.02,curves=preset=increase_contrast",
    "neon": "eq=contrast=1.20:saturation=1.55:brightness=-0.02,curves=preset=strong_contrast",
    "dream": "gblur=sigma=1.6,eq=contrast=0.98:saturation=1.15:brightness=0.03",
    "retro": "curves=preset=vintage,eq=contrast=1.05:saturation=1.10:gamma_g=1.05",
    "moody": "eq=contrast=1.18:saturation=0.90:brightness=-0.04,"
             "curves=r='0/0.02 0.5/0.42 1/0.90':b='0/0.05 0.5/0.5 1/0.92'",
}
COLOR_LABELS = {
    "none": "Naturale", "vivid": "Vivido", "cinematic": "Cinematico", "vhs": "VHS / vintage",
    "warm": "Caldo (golden hour)", "cold": "Freddo (teal)", "film": "Pellicola sbiadita",
    "bw": "Bianco e nero", "neon": "Neon (contrasto forte)", "dream": "Sognante (soft)",
    "retro": "Retro anni '70", "moody": "Cupo",
}
# grade che stanno bene mischiati quando "varieta'" e' alta
_VARIETY_GRADES = ["vivid", "warm", "cold", "neon", "moody", "film", "retro", "cinematic"]


def pick_color_style(rng: random.Random, base: str, variety: float) -> str:
    if variety <= 0.0 or rng.random() >= variety:
        return base
    pool = [g for g in _VARIETY_GRADES if g != base] or _VARIETY_GRADES
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Rifiniture globali (uguali su tutti i segmenti).
# ---------------------------------------------------------------------------
def finish_chain(grain: float = 0.0, vignette: bool = False, chromatic: bool = False) -> str:
    parts = []
    if chromatic:
        parts.append("rgbashift=rh=2:bh=-2")
    if vignette:
        parts.append("vignette=PI/4.5")
    if grain and grain > 0:
        parts.append(f"noise=alls={max(1, int(grain * 22))}:allf=t")
    return ",".join(parts)


# ---------------------------------------------------------------------------
# Effetti d'impatto: brevi, applicati ai primi frame dei tagli accentati.
# ---------------------------------------------------------------------------
IMPACT_EFFECTS = {
    "flash": "eq=brightness=0.65:contrast=1.15:enable='lt(t,0.05)'",
    "flashblack": "eq=brightness=-0.85:enable='lt(t,0.045)'",
    "rgbsplit": "rgbashift=rh=10:bh=-10:gv=6:enable='lt(t,0.13)'",
    "blur": "gblur=sigma=14:enable='lt(t,0.10)'",
    "shake": ("crop=w=iw-72:h=ih-72:"
              "x='36+sin(t*95)*30*max(0\\,1-t/0.30)':"
              "y='36+cos(t*82)*24*max(0\\,1-t/0.30)',scale=iw+72:ih+72"),
    "pulse": "eq=saturation=1.6:contrast=1.12:enable='lt(t,0.08)'",
    "glitch": ("rgbashift=rh=16:bh=-16:gv=10:enable='lt(t,0.11)',"
               "noise=alls=44:allf=t:enable='lt(t,0.08)'"),
    "whip": ("crop=w=iw-160:h=ih:x='(iw-160)*(1-min(1\\,t/0.11))':y=0,scale=iw+160:ih,"
             "gblur=sigma=26:enable='lt(t,0.09)'"),
}
IMPACT_LABELS = {
    "flash": "Flash bianco", "flashblack": "Flash nero", "rgbsplit": "RGB split",
    "blur": "Blur punch", "shake": "Shake", "pulse": "Pulse colore",
    "glitch": "Glitch digitale", "whip": "Whip pan",
}


def pick_impact(rng: random.Random, enabled) -> str:
    """Ritorna la CHIAVE di un effetto d'impatto a caso fra quelli
    abilitati (o "" se nessuno)."""
    pool = [e for e in (enabled or []) if e in IMPACT_EFFECTS]
    return rng.choice(pool) if pool else ""


def pick_no_repeat(pool, rng: random.Random, avoid=None):
    """Sceglie da `pool` evitando `avoid` (l'ultimo valore usato), cosi'
    lo stesso effetto/movimento/transizione non esce due volte di fila."""
    pool = list(pool or [])
    if not pool:
        return None
    choices = [p for p in pool if p != avoid] or pool
    return rng.choice(choices)


# ---------------------------------------------------------------------------
# Stili di montaggio ("ricette"): pacchetti coerenti di comportamento, per
# avvicinarsi ai preset TikTok/CapCut invece di randomizzare a caso.
#   xfade_ms      durata transizione (0 = stacchi netti sul beat)
#   xfade_pool    nomi xfade quando non e' hard-cut (None = logica di default)
#   motions       da cui pescare il movimento per ogni clip (anti-ripetizione)
#   impacts       effetti d'impatto disponibili
#   impact_on     "hits" (solo i colpi forti) | "downbeats" | "off"
#   impact_min_gap  secondi minimi fra un effetto e il successivo
#   *_intensity / grain / vignette / chromatic / variety  -> rifiniture
# ---------------------------------------------------------------------------
#   video_speeds  pool (con ripetizioni = pesi) per la velocita' delle clip video
#   hold_prob     probabilita' che un segmento sia "hero" e duri 2 beat
EDIT_STYLES = {
    "clean": {
        "xfade_ms": 0, "xfade_pool": None,
        "motions": ["bounce", "punch", "kenburns", "static"],
        "impacts": ["flash"], "impact_on": "hits", "impact_min_gap": 2.2,
        "grain": 0.0, "vignette": False, "chromatic": False,
        "motion_intensity": 1.0, "variety": 0.4,
        "video_speeds": [1.0, 1.0, 1.0, 0.5], "hold_prob": 0.18,
    },
    "hype": {
        "xfade_ms": 0, "xfade_pool": None,
        "motions": ["punch", "bounce", "handheld"],
        "impacts": ["shake", "rgbsplit", "flash", "flashblack", "whip"],
        "impact_on": "downbeats", "impact_min_gap": 0.9,
        "grain": 0.08, "vignette": True, "chromatic": True,
        "motion_intensity": 1.5, "variety": 0.7,
        "video_speeds": [1.0, 1.0, 2.0, 1.6, 0.5], "hold_prob": 0.10,
    },
    "smooth": {
        "xfade_ms": 320, "xfade_pool": ["fade", "smoothleft", "smoothright", "circleopen", "dissolve"],
        "motions": ["kenburns", "sway", "zoom"],
        "impacts": [], "impact_on": "off", "impact_min_gap": 99.0,
        "grain": 0.05, "vignette": True, "chromatic": False,
        "motion_intensity": 0.75, "variety": 0.25,
        "video_speeds": [1.0, 0.5, 0.5, 1.0], "hold_prob": 0.30,
    },
    "glitch": {
        "xfade_ms": 0, "xfade_pool": None,
        "motions": ["punch", "static", "handheld"],
        "impacts": ["glitch", "rgbsplit", "flashblack", "blur"],
        "impact_on": "downbeats", "impact_min_gap": 1.0,
        "grain": 0.2, "vignette": False, "chromatic": True,
        "motion_intensity": 1.2, "variety": 0.6,
        "video_speeds": [1.0, 1.0, 2.0, 0.5], "hold_prob": 0.12,
    },
    "retro": {
        "xfade_ms": 200, "xfade_pool": ["fade", "fadeblack", "dissolve", "pixelize"],
        "motions": ["sway", "kenburns", "static"],
        "impacts": ["flashblack", "rgbsplit"], "impact_on": "hits", "impact_min_gap": 2.5,
        "grain": 0.5, "vignette": True, "chromatic": True,
        "motion_intensity": 0.9, "variety": 0.35,
        "video_speeds": [1.0, 0.5, 1.0], "hold_prob": 0.22,
    },
}
EDIT_STYLE_LABELS = {
    "custom": "Personalizzato (usa il pannello Effetti)",
    "clean": "Pulito — stacchi netti sul beat",
    "hype": "Hype — punch, shake, RGB, veloce",
    "smooth": "Morbido — transizioni + Ken Burns",
    "glitch": "Glitch — RGB split, disturbo",
    "retro": "Retro — pellicola, grana",
}


# ---------------------------------------------------------------------------
# Transizioni xfade.
# ---------------------------------------------------------------------------
NORMAL_TRANSITIONS = [
    "zoomin", "slideleft", "slideright", "circleopen",
    "smoothleft", "smoothright", "wiperight", "distance",
]
ACCENT_TRANSITIONS = [
    "fadewhite", "pixelize", "hlslice", "vuslice", "hrslice", "radial",
]
TRANSITION_MODES = {
    "fade": ["fade", "fadeblack", "fadewhite"],
    "slide": ["slideleft", "slideright", "slideup", "slidedown", "smoothleft", "smoothright"],
    "zoom": ["zoomin"],
}


def pick_transition(rng: random.Random, accented: bool, mode: str = "auto") -> str:
    if mode == "cut":
        return "fade"                      # la durata viene forzata a ~0 altrove
    if mode == "chaos":
        return rng.choice(NORMAL_TRANSITIONS + ACCENT_TRANSITIONS)
    if mode in TRANSITION_MODES:
        return rng.choice(TRANSITION_MODES[mode])
    pool = ACCENT_TRANSITIONS if accented else NORMAL_TRANSITIONS
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Movimento (Ken Burns e varianti).
# ---------------------------------------------------------------------------
MOTION_MODES = ["kenburns", "zoom", "sway", "handheld", "punch", "bounce", "static"]
MOTION_LABELS = {
    "kenburns": "Ken Burns (zoom + panoramica)",
    "zoom": "Zoom pulito",
    "sway": "Dondolio leggero",
    "handheld": "Camera a mano (mosso)",
    "punch": "Punch ritmico",
    "bounce": "Bounce sul beat (stile CapCut)",
    "static": "Fermo (nessun movimento)",
}
_VARIETY_MOTIONS = ["kenburns", "zoom", "sway", "handheld", "bounce"]


def pick_motion(rng: random.Random, base: str, variety: float) -> str:
    if variety <= 0.0 or rng.random() >= variety * 0.7:
        return base
    pool = [m for m in _VARIETY_MOTIONS if m != base] or _VARIETY_MOTIONS
    return rng.choice(pool)


def crop_to_fill(src_w: int, src_h: int, target_w: int, target_h: int,
                  zoom: float = 1.0, ax: float = 0.5, ay: float = 0.5) -> str:
    """scale+crop che riempie il frame target senza deformare.

    `zoom` >= 1 stringe l'inquadratura (piu' alto = piu' vicino). `ax`/`ay`
    in [0,1] spostano il ritaglio: 0.5 = centro, 0 = alto/sinistra,
    1 = basso/destra. Con zoom=1 e ax=ay=0.5 e' il vecchio crop centrato.
    """
    z = max(1.0, float(zoom))
    ax = min(1.0, max(0.0, float(ax)))
    ay = min(1.0, max(0.0, float(ay)))
    sw, sh = int(round(target_w * z)), int(round(target_h * z))
    return (
        f"scale={sw}:{sh}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h}:x=(iw-ow)*{ax:.4f}:y=(ih-oh)*{ay:.4f}"
    )


def video_motion_filter(target_w: int, target_h: int, fps: int, rng: random.Random,
                         punch: bool = False, strong: bool = False, intensity: float = 1.0) -> str:
    """Movimento LEGGERO per le clip video: un lentissimo zoom di deriva
    piu' un eventuale colpo di zoom iniziale sui tagli accentati. Usa
    `zoompan` con d=1 (un frame di uscita per frame di ingresso: non
    altera il tempo del video)."""
    s = max(0.3, min(2.0, float(intensity)))
    drift = rng.uniform(0.02, 0.05) * s
    zbase = f"1.0+{drift:.4f}*min(1,on/({fps}*2.5))"     # deriva fino a ~drift in ~2.5s
    if punch:
        amp = (0.11 if strong else 0.06) * min(1.4, s)
        span = 7 if strong else 5
        zexpr = f"if(lte(on,{span}),1.0+{amp:.3f}*sin(on/{span}*PI),{zbase})"
    else:
        zexpr = zbase
    return (
        f"scale=iw*1.16:ih*1.16,"
        f"zoompan=z='{zexpr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={target_w}x{target_h}:fps={fps}"
    )


def kenburns_filter(
    target_w: int,
    target_h: int,
    duration: float,
    fps: int,
    rng: random.Random,
    punch_in: bool = False,
    strong: bool = False,
    motion: str = "kenburns",
    intensity: float = 1.0,
) -> str:
    """Catena scale+zoompan per il movimento della clip.

    `motion` sceglie il tipo (vedi MOTION_MODES), `intensity` (0.2..2.5)
    scala ampiezza di zoom e jitter. `punch_in`/`strong` aggiungono un
    colpo di zoom iniziale sui tagli accentati / sul downbeat.
    """
    frames = max(2, int(duration * fps))
    s = max(0.2, min(2.5, float(intensity)))

    if motion == "static":
        z = "1.0"
        if punch_in:
            amp = (0.16 if strong else 0.09) * min(1.5, s)
            span = 8 if strong else 6
            z = f"if(lte(on,{span}),1.0+{amp:.3f}*sin(on/{span}*PI),1.0)"
        return (f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={target_w}x{target_h}:fps={fps}")

    zoom_target = min(1.6, 1.0 + (rng.uniform(1.12, 1.30) - 1.0) * s)
    zoom_speed = (zoom_target - 1.0) / frames

    jx = rng.uniform(0.004, 0.012) * s
    jy = rng.uniform(0.004, 0.010) * s
    jf = rng.uniform(0.2, 0.9)
    ph = rng.uniform(0, 6.28)
    if motion == "handheld":
        jx *= 2.6
        jy *= 2.6
    elif motion in ("zoom", "punch", "bounce"):
        jx = jy = 0.0
    elif motion == "sway":
        jx *= 1.4

    jitx = f"+(iw*{jx:.4f}*sin(on*{jf:.3f}+{ph:.3f}))" if jx else ""
    jity = f"+(ih*{jy:.4f}*cos(on*{jf:.3f}+{ph:.3f}))" if jy else ""

    if motion == "sway":
        drift = rng.choice([-1.0, 1.0]) * 0.06 * s
        zexpr = "1.06"
        xexpr = f"iw/2-(iw/zoom/2)+(iw*{drift:.4f}*on/{frames}){jitx}"
        yexpr = f"ih/2-(ih/zoom/2){jity}"
    elif motion == "punch":
        period = max(6, int(fps * 0.42))
        base = 1.04 + 0.03 * s
        amp = 0.05 * s
        zexpr = f"{base:.3f}+{amp:.3f}*abs(sin(on/{period}*PI))"
        xexpr = f"iw/2-(iw/zoom/2){jitx}"
        yexpr = f"ih/2-(ih/zoom/2){jity}"
    elif motion == "bounce":
        # "colpo" di zoom nei primi 3 frame, poi si assesta: il look da
        # montaggio-foto CapCut, un rimbalzo per taglio, senza deriva.
        hit = 1.09 + 0.10 * s
        settle = 1.04 + 0.03 * s
        rise = f"{settle:.3f}+({hit - settle:.3f})*(on/3)"
        fall = f"{hit:.3f}-({hit - settle:.3f})*min(1,(on-3)/12)"
        zexpr = f"if(lte(on,3),{rise},{fall})"
        xexpr = "iw/2-(iw/zoom/2)"
        yexpr = "ih/2-(ih/zoom/2)"
    else:
        dirs = ["in", "out", "pan_l", "pan_r"] if motion == "kenburns" else ["in", "out"]
        direction = rng.choice(dirs)
        if direction == "in":
            zexpr = f"min(zoom+{zoom_speed:.5f},{zoom_target:.3f})"
            xexpr = f"iw/2-(iw/zoom/2){jitx}"
            yexpr = f"ih/2-(ih/zoom/2){jity}"
        elif direction == "out":
            zexpr = f"if(eq(on,1),{zoom_target:.3f},max(zoom-{zoom_speed:.5f},1.0))"
            xexpr = f"iw/2-(iw/zoom/2){jitx}"
            yexpr = f"ih/2-(ih/zoom/2){jity}"
        elif direction == "pan_l":
            zexpr = f"min(zoom+{zoom_speed*0.7:.5f},{min(zoom_target,1.22):.3f})"
            xexpr = f"iw/2-(iw/zoom/2)+(iw*0.05*on/{frames}){jitx}"
            yexpr = f"ih/2-(ih/zoom/2){jity}"
        else:
            zexpr = f"min(zoom+{zoom_speed*0.7:.5f},{min(zoom_target,1.22):.3f})"
            xexpr = f"iw/2-(iw/zoom/2)-(iw*0.05*on/{frames}){jitx}"
            yexpr = f"ih/2-(ih/zoom/2){jity}"

    if punch_in:
        amp = (0.16 if strong else 0.09) * min(1.5, s)
        span = 8 if strong else 6
        zexpr = f"if(lte(on,{span}),1.0+{amp:.3f}*sin(on/{span}*PI),{zexpr})"

    return (
        f"scale=iw*2:ih*2,"
        f"zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}':d={frames}:"
        f"s={target_w}x{target_h}:fps={fps}"
    )
