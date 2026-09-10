"""Analisi pose per i reel-tutorial di pole dance / exotic.

Cosa fa:
- keypoint del corpo per fotogramma (MediaPipe Pole Landmarker, Tasks API)
- posizione del PALO (linea quasi verticale, Hough)
- CONTATTI: quando una mano / un piede e' sul palo e regge
- FERMI ("hold"): tratti a bassissimo movimento -> momenti da congelare,
  con la parte del corpo in FOCUS (la presa) e il riquadro per lo zoom

Il nome della mossa e il consiglio all'allieva li mette poi `vision.py`
(modello multimodale locale) sui pochi fotogrammi di focus.

CLI:  python -m autoedit.pose --video clip.mp4 --out preview.mp4
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent / "assets_data" / "models"
_POSE_TASK = _MODEL_DIR / "pose_landmarker_full.task"
_POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

# indici landmark MediaPipe Pose (33 punti)
IDX = {
    "nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_index": 19, "r_index": 20, "l_thumb": 21, "r_thumb": 22,
    "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30, "l_foot": 31, "r_foot": 32,
}
_SKELETON = [
    ("l_shoulder", "r_shoulder"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_hip", "r_hip"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"), ("l_ankle", "l_foot"), ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"), ("r_ankle", "r_foot"),
]
# parte "presa" -> landmark che la rappresentano
GRIP_PARTS = {
    "left_hand": ("l_wrist", "l_index", "l_thumb"),
    "right_hand": ("r_wrist", "r_index", "r_thumb"),
    "left_foot": ("l_ankle", "l_heel", "l_foot"),
    "right_foot": ("r_ankle", "r_heel", "r_foot"),
}
PART_IT = {"left_hand": "mano sx", "right_hand": "mano dx",
           "left_foot": "piede sx", "right_foot": "piede dx"}


def ensure_model() -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not _POSE_TASK.is_file() or _POSE_TASK.stat().st_size < 100_000:
        urllib.request.urlretrieve(_POSE_URL, _POSE_TASK)
    return _POSE_TASK


# --------------------------------------------------------------------------
@dataclass
class PoseFrame:
    t: float                                   # secondi
    lm: Optional[np.ndarray] = None            # (33, 3): x, y in [0,1], visibility
    n_people: int = 0


@dataclass
class Contact:
    part: str                                  # left_hand / right_hand / left_foot / right_foot
    t0: float
    t1: float
    px: tuple = (0.0, 0.0)                      # posizione media in pixel

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


@dataclass
class Hold:
    t0: float
    t1: float
    focus_t: float                             # fotogramma su cui congelare
    focus_part: str                            # parte del corpo in focus (la presa)
    bbox: tuple = (0.0, 0.0, 1.0, 1.0)         # x0,y0,x1,y1 normalizzati, per lo zoom
    contacts: list = field(default_factory=list)

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


@dataclass
class PoseEvent:
    """Momento 'scenico' leggibile dalla geometria dei keypoint:
    inversione del corpo o estensione massima di un arto."""
    kind: str                                  # "invert" | "arm_ext" | "leg_ext" | "straddle" | "line"
    t: float
    part: str = ""                             # arto interessato (per *_ext)
    value: float = 0.0                         # angolo / punteggio
    xy: tuple = (0.5, 0.5)                      # punto su cui puntare (endpoint dell'arto)
    label: str = ""


# --------------------------------------------------------------------------
def analyze_video(path: Path | str, fps_sample: float = 8.0, max_people: int = 1,
                  t_start: float = 0.0, t_end: Optional[float] = None) -> list[PoseFrame]:
    """Campiona il video a ~`fps_sample` e ritorna i keypoint per frame.
    Se dati, analizza solo la finestra [t_start, t_end] secondi."""
    import cv2  # noqa: PLC0415
    import mediapipe as mp  # noqa: PLC0415
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )

    ensure_model()
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_POSE_TASK)),
        running_mode=RunningMode.VIDEO, num_poses=max(1, int(max_people)),
        min_pose_detection_confidence=0.4, min_tracking_confidence=0.4,
    )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(1.0, fps_sample))))
    i0 = int(max(0.0, t_start) * src_fps)
    i1 = int(t_end * src_fps) if t_end else 1 << 62
    if i0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)

    frames: list[PoseFrame] = []
    with PoseLandmarker.create_from_options(opts) as lm:
        i = i0
        while i <= i1:
            ok, bgr = cap.read()
            if not ok:
                break
            if (i - i0) % step == 0:
                t = i / src_fps
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                    int(t * 1000))
                pf = PoseFrame(t=t, n_people=len(res.pose_landmarks))
                if res.pose_landmarks:
                    p = res.pose_landmarks[0]
                    pf.lm = np.array([[k.x, k.y, getattr(k, "visibility", 1.0)] for k in p],
                                     dtype=np.float32)
                frames.append(pf)
            i += 1
    cap.release()
    return frames


def pole_x(path: Path | str, probe_frames: int = 12) -> Optional[float]:
    """x NORMALIZZATA (0..1) del palo: linea quasi verticale piu' votata,
    mediata su alcuni fotogrammi. None se non trovata."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None
    xs: list[float] = []
    for f in np.linspace(total * 0.08, total * 0.92, probe_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 140)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=int(h * 0.35), maxLineGap=40)
        if lines is None:
            continue
        cand = []
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang > 74:                                 # quasi verticale
                cand.append((x1 + x2) / 2.0 / w)
        if cand:
            xs.append(float(np.median(cand)))
    cap.release()
    if len(xs) < max(2, probe_frames // 4):              # troppo poche linee -> non fidarsi
        return None
    xs = np.array(xs)
    # tieni il gruppo piu' coerente (il palo e' fermo, il resto e' rumore)
    med = float(np.median(xs))
    keep = xs[np.abs(xs - med) < 0.06]
    return float(np.median(keep)) if len(keep) >= 2 else None


def pole_x_auto(path: Path | str, frames: list[PoseFrame]) -> tuple[Optional[float], str]:
    """x del palo, robusta: nel pole la presa e' SUL palo, quindi i
    keypoint sono il segnale piu' affidabile. Hough solo di conferma /
    ripiego. Ritorna (x | None, sorgente)."""
    kp = pole_x_from_pose(frames)
    hg = pole_x(path)
    if kp is not None and hg is not None and abs(kp - hg) < 0.08:
        return (kp + hg) / 2, "keypoint+Hough"
    if kp is not None:
        return kp, "keypoint"
    if hg is not None:
        return hg, "Hough"
    return None, "non trovato"


def pole_x_from_pose(frames: list[PoseFrame]) -> Optional[float]:
    """Stima il palo DAL corpo: nel pole la presa (polsi/caviglie) sta
    quasi sempre incolonnata su una x. Utile quando Hough fallisce
    (palo cromato, sfondo confuso, angolo di camera)."""
    xs: list[float] = []
    for pf in frames:
        if pf.lm is None:
            continue
        for n in ("l_wrist", "r_wrist", "l_ankle", "r_ankle"):
            k = pf.lm[IDX[n]]
            if k[2] >= 0.5:
                xs.append(float(k[0]))
    if len(xs) < 8:
        return None
    xs = np.array(xs)
    # moda robusta: centro della finestra 0.12 piu' popolata
    grid = np.linspace(0.1, 0.9, 33)
    best = grid[np.argmax([np.sum(np.abs(xs - g) < 0.06) for g in grid])]
    inl = xs[np.abs(xs - best) < 0.06]
    return float(np.median(inl)) if len(inl) >= 6 else None


def _part_xy(lm: np.ndarray, names, vis_min: float = 0.4):
    pts = [lm[IDX[n]] for n in names if lm[IDX[n], 2] >= vis_min]
    if not pts:
        return None
    a = np.mean(pts, axis=0)
    return float(a[0]), float(a[1])


def contacts(frames: list[PoseFrame], px_pole: Optional[float], band: float = 0.09,
             min_dur: float = 0.25) -> list[Contact]:
    """Intervalli in cui una mano/piede sta sul palo (entro `band` in x) e
    si muove poco. `px_pole` normalizzato; se None, niente contatti."""
    if px_pole is None:
        return []
    out: list[Contact] = []
    for part, names in GRIP_PARTS.items():
        run_start = None
        last_xy = None
        acc = []
        for pf in frames:
            near = False
            xy = None
            if pf.lm is not None:
                xy = _part_xy(pf.lm, names)
                if xy is not None:
                    slow = last_xy is None or (abs(xy[0] - last_xy[0]) + abs(xy[1] - last_xy[1])) < 0.06
                    near = abs(xy[0] - px_pole) < band and slow
            if near:
                if run_start is None:
                    run_start = pf.t
                    acc = []
                acc.append(xy)
            else:
                if run_start is not None and pf.t - run_start >= min_dur:
                    m = np.mean(acc, axis=0)
                    out.append(Contact(part, run_start, pf.t, (float(m[0]), float(m[1]))))
                run_start = None
            last_xy = xy
        if run_start is not None and frames and frames[-1].t - run_start >= min_dur:
            m = np.mean(acc, axis=0) if acc else (px_pole, 0.5)
            out.append(Contact(part, run_start, frames[-1].t, (float(m[0]), float(m[1]))))
    return sorted(out, key=lambda c: c.t0)


def _motion(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 1.0
    m = (a[:, 2] > 0.4) & (b[:, 2] > 0.4)
    if m.sum() < 6:
        return 1.0
    return float(np.mean(np.linalg.norm(a[m, :2] - b[m, :2], axis=1)))


def _smooth(frames: list[PoseFrame], k: int = 2) -> list[np.ndarray | None]:
    """Mediana mobile sui keypoint: toglie il jitter di MediaPipe che
    altrimenti fa sembrare 'in movimento' anche una posa ferma."""
    out: list[np.ndarray | None] = []
    for i, pf in enumerate(frames):
        if pf.lm is None:
            out.append(None)
            continue
        win = [f.lm for f in frames[max(0, i - k): i + k + 1] if f.lm is not None]
        out.append(np.median(np.stack(win), axis=0) if len(win) >= 2 else pf.lm)
    return out


def detect_holds(frames: list[PoseFrame], cts: list[Contact],
                 still: float = 0.012, min_hold: float = 0.6) -> list[Hold]:
    """Tratti fermi (movimento medio dei keypoint < `still`) lunghi almeno
    `min_hold` s. Il FOCUS e' il contatto iniziato piu' di recente.
    I keypoint vengono prima lisciati per togliere il jitter."""
    sm = _smooth(frames)
    holds: list[Hold] = []
    run_start = None
    lowest = (1e9, 0.0)
    for i in range(1, len(frames)):
        mv = _motion(sm[i - 1], sm[i])
        t = frames[i].t
        if mv < still and frames[i].lm is not None:
            if run_start is None:
                run_start = frames[i - 1].t
                lowest = (mv, t)
            elif mv < lowest[0]:
                lowest = (mv, t)
        else:
            if run_start is not None and t - run_start >= min_hold:
                holds.append(_finish_hold(run_start, t, lowest[1], frames, cts))
            run_start = None
    if run_start is not None and frames and frames[-1].t - run_start >= min_hold:
        holds.append(_finish_hold(run_start, frames[-1].t, lowest[1], frames, cts))
    return holds


def _finish_hold(t0, t1, focus_t, frames, cts) -> Hold:
    active = [c for c in cts if c.t0 <= t1 and c.t1 >= t0]
    focus = max(active, key=lambda c: c.t0) if active else None
    part = focus.part if focus else ""
    fr = min(frames, key=lambda f: abs(f.t - focus_t))
    bbox = (0.0, 0.0, 1.0, 1.0)
    if focus and fr.lm is not None:
        xy = _part_xy(fr.lm, GRIP_PARTS[part]) or focus.px
        r = 0.16
        bbox = (max(0, xy[0] - r), max(0, xy[1] - r), min(1, xy[0] + r), min(1, xy[1] + r))
    return Hold(t0, t1, focus_t, part, bbox, active)


# --------------------------------------------------------------------------
# Eventi "scenici": inversione del corpo, estensione massima di un arto
# --------------------------------------------------------------------------
def _ang(a, b, c) -> float:
    """Angolo in gradi al vertice b, fra i segmenti b->a e b->c."""
    ba, bc = np.asarray(a[:2]) - b[:2], np.asarray(c[:2]) - b[:2]
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nc), -1, 1))))


_LIMBS = {
    "braccio sx": ("l_shoulder", "l_elbow", "l_wrist"),
    "braccio dx": ("r_shoulder", "r_elbow", "r_wrist"),
    "gamba sx": ("l_hip", "l_knee", "l_ankle"),
    "gamba dx": ("r_hip", "r_knee", "r_ankle"),
}
_LIMB_END = {"braccio sx": "l_wrist", "braccio dx": "r_wrist",
             "gamba sx": "l_ankle", "gamba dx": "r_ankle"}


def body_metrics(lm: np.ndarray) -> dict:
    """Misure geometriche istantanee da un set di keypoint."""
    def g(n):
        return lm[IDX[n]]

    sh = (g("l_shoulder")[:2] + g("r_shoulder")[:2]) / 2
    hp = (g("l_hip")[:2] + g("r_hip")[:2]) / 2
    torso = hp - sh                                  # spalle -> fianchi
    # angolo del busto rispetto alla verticale "in giu'" (0 = in piedi, +-180 = a testa in giu')
    tilt = float(np.degrees(np.arctan2(torso[0], torso[1])))
    # "sottosopra" = busto oltre l'orizzontale verso l'alto. `tilt` regge meglio
    # del confronto fianchi/spalle quando i keypoint sono rumorosi.
    inverted = bool(abs(tilt) > 115.0 or hp[1] < sh[1] - 0.03)
    m = {"torso_tilt": tilt, "inverted": inverted,
         "straddle": _ang(g("l_ankle"), np.append(hp, 1.0), g("r_ankle"))}
    for name, (a, b, c) in _LIMBS.items():
        if min(g(a)[2], g(b)[2], g(c)[2]) >= 0.4:
            m[name] = _ang(g(a), g(b), g(c))
    return m


def detect_events(frames: list[PoseFrame], ext_min: float = 165.0,
                  straddle_min: float = 150.0) -> list[PoseEvent]:
    """Trova: inversioni (busto a testa in giu') e i PICCHI di estensione
    di braccia/gambe (arto quasi dritto -> angolo ~180 in un massimo locale)."""
    ev: list[PoseEvent] = []
    sm = _smooth(frames)
    metr = [body_metrics(l) if l is not None else None for l in sm]

    # --- inversioni: run contigui (min 2 frame) ---
    run = None
    for i, mm in enumerate(metr):
        inv = mm is not None and mm["inverted"]
        if inv and run is None:
            run = i
        elif not inv and run is not None:
            if i - run < 2:                          # lampo isolato: rumore, ignora
                run = None
                continue
            j = max(range(run, i), key=lambda k: abs(metr[k]["torso_tilt"]))
            mm2 = metr[j]
            ev.append(PoseEvent("invert", frames[j].t, value=mm2["torso_tilt"],
                                xy=tuple(((sm[j][IDX["l_hip"]][:2] + sm[j][IDX["r_hip"]][:2]) / 2)),
                                label="A TESTA IN GIU'"))
            run = None
    if run is not None:
        j = max(range(run, len(metr)), key=lambda k: abs(metr[k]["torso_tilt"]))
        ev.append(PoseEvent("invert", frames[j].t, value=metr[j]["torso_tilt"],
                            xy=tuple(((sm[j][IDX["l_hip"]][:2] + sm[j][IDX["r_hip"]][:2]) / 2)),
                            label="A TESTA IN GIU'"))

    # --- picchi di estensione per ogni arto ---
    # serve PROMINENZA: l'arto deve essersi prima piegato (< thresh-flex) e poi
    # essersi disteso; altrimenti un arto sempre dritto spara eventi ad ogni frame.
    def peaks(series, times, thresh, kind, part, flex=28.0):
        last = -9.0
        min_since = 999.0
        for i in range(1, len(series) - 1):
            a, b, c = series[i - 1], series[i], series[i + 1]
            if b is None:
                continue
            min_since = min(min_since, b)
            if a is None or c is None:
                continue
            local_max = b >= a and b >= c
            if (local_max and b >= thresh and (thresh - min_since) >= flex
                    and times[i] - last > 0.6):
                last = times[i]
                min_since = b
                yield PoseEvent(kind, times[i], part=part, value=b, label=f"MAX EST · {part}")

    T = [f.t for f in frames]
    for part in _LIMBS:
        s = [mm.get(part) if mm else None for mm in metr]
        for e in peaks(s, T, ext_min, "arm_ext" if "braccio" in part else "leg_ext", part):
            k = min(range(len(frames)), key=lambda x: abs(frames[x].t - e.t))
            if sm[k] is not None:
                e.xy = tuple(sm[k][IDX[_LIMB_END[part]]][:2])
            ev.append(e)
    sstr = [mm.get("straddle") if mm else None for mm in metr]
    for e in peaks(sstr, T, straddle_min, "straddle", "gambe"):
        e.label = "APERTURA MAX"
        ev.append(e)
    return sorted(ev, key=lambda x: x.t)


# --------------------------------------------------------------------------
def annotate(path: Path | str, out_path: Path | str, frames: list[PoseFrame],
             holds: list[Hold], px_pole: Optional[float]) -> Path:
    """Video di anteprima: scheletro + palo + contatti + barra dei fermi."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_t = sorted(frames, key=lambda f: f.t)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / fps if total else (by_t[-1].t if by_t else 1.0)

    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = i / fps
        pf = min(by_t, key=lambda f: abs(f.t - t)) if by_t else None
        if px_pole is not None:
            cv2.line(bgr, (int(px_pole * w), 0), (int(px_pole * w), h), (0, 180, 255), 2)
        if pf is not None and pf.lm is not None:
            L = pf.lm
            for a, b in _SKELETON:
                pa, pb = L[IDX[a]], L[IDX[b]]
                if pa[2] > 0.3 and pb[2] > 0.3:
                    cv2.line(bgr, (int(pa[0] * w), int(pa[1] * h)),
                             (int(pb[0] * w), int(pb[1] * h)), (60, 230, 90), 2)
            for part, names in GRIP_PARTS.items():
                xy = _part_xy(L, names, 0.3)
                if xy:
                    on = px_pole is not None and abs(xy[0] - px_pole) < 0.09
                    cv2.circle(bgr, (int(xy[0] * w), int(xy[1] * h)), 9,
                               (0, 90, 255) if not on else (0, 255, 120), -1 if on else 2)
        hold_now = next((hd for hd in holds if hd.t0 <= t <= hd.t1), None)
        if hold_now:
            cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (0, 255, 120), 6)
            cv2.putText(bgr, f"FOCUS: {PART_IT.get(hold_now.focus_part, hold_now.focus_part or '?')}",
                        (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 120), 3)
        # barra dei fermi in basso
        y = h - 14
        cv2.rectangle(bgr, (0, y - 6), (w, h), (30, 30, 30), -1)
        for hd in holds:
            x0 = int(hd.t0 / dur * w)
            x1 = int(hd.t1 / dur * w)
            cv2.rectangle(bgr, (x0, y - 6), (x1, h), (0, 255, 120), -1)
        cv2.line(bgr, (int(t / dur * w), y - 8), (int(t / dur * w), h), (255, 255, 255), 2)
        vw.write(bgr)
        i += 1
    cap.release()
    vw.release()
    return Path(out_path)


def _bbox_visible(lm: np.ndarray, vis_min: float = 0.4):
    m = lm[:, 2] >= vis_min
    if m.sum() < 4:
        return None
    xs, ys = lm[m, 0], lm[m, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _torso_bbox(lm: np.ndarray):
    """Riquadro attorno al busto (spalle + fianchi), per il fuoco di
    default quando non c'e' una presa specifica."""
    names = ["l_shoulder", "r_shoulder", "l_hip", "r_hip"]
    pts = [lm[IDX[n]] for n in names if lm[IDX[n], 2] >= 0.3]
    if len(pts) < 3:
        return _bbox_visible(lm)
    a = np.array(pts)
    cx, cy = float(a[:, 0].mean()), float(a[:, 1].mean())
    r = min(0.24, max(0.13, float(np.ptp(a[:, 0])) * 0.7, float(np.ptp(a[:, 1])) * 0.6))
    return (cx - r, cy - r, cx + r, cy + r)


def debug_video(path: Path | str, out_path: Path | str, frames: list[PoseFrame],
                holds: list[Hold], cts: list[Contact], px_pole: Optional[float],
                labels: Optional[dict] = None, events: Optional[list] = None,
                t_start: float = 0.0, t_end: Optional[float] = None,
                max_w: int = 1280, transcode: bool = True) -> Path:
    """Video DEBUG: guardi attraverso un mirino da reflex e vedi DOVE
    l'IA sta mettendo il fuoco (staffe AF che scattano sulla presa /
    sul soggetto), la griglia dei punti AF, il palo, lo scheletro, e un
    HUD con parte in focus / nome mossa / FOCUS LOCK sui fermi.

    Renderizza solo [t_start, t_end] e riscala a `max_w` di larghezza
    (una clip lunga in 1080p e' pesantissima da disegnare frame per frame).
    `labels` = {indice_hold: {"move": str, "grip": str}} da vision.py.
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / fps if total else (frames[-1].t if frames else 1.0)
    scale = min(1.0, max_w / sw) if sw else 1.0
    w, h = int(sw * scale), int(sh * scale)
    i0 = int(max(0.0, t_start) * fps)
    i1 = int(t_end * fps) if t_end else (total or 1 << 62)
    if i0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    by_t = sorted(frames, key=lambda f: f.t)
    labels = labels or {}
    events = events or []
    EV_COL = {"invert": (240, 80, 240), "arm_ext": (255, 220, 40),
              "leg_ext": (255, 220, 40), "straddle": (255, 140, 40)}

    raw = Path(out_path).with_suffix(".raw.mp4") if transcode else Path(out_path)
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError("cv2.VideoWriter non si apre (codec mp4v mancante?).")

    GREEN, AMBER, DIM = (90, 255, 120), (60, 200, 255), (120, 150, 120)
    FT = cv2.FONT_HERSHEY_DUPLEX
    m = int(min(w, h) * 0.045)                       # margine mirino
    af_cols, af_rows = 7, 5
    # box di fuoco "smussato": segue con inerzia il target
    fx = [w * 0.5, h * 0.5, w * 0.28, h * 0.28]      # cx, cy, half-w, half-h

    def bracket(img, cx, cy, hw, hh, col, thick, ln):
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = int(cx + sx * hw), int(cy + sy * hh)
                cv2.line(img, (x, y), (int(x - sx * ln), y), col, thick)
                cv2.line(img, (x, y), (x, int(y - sy * ln)), col, thick)

    i = i0
    while i <= i1:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if scale < 1.0:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        t = (i - 1) / fps
        pf = min(by_t, key=lambda f: abs(f.t - t)) if by_t else None
        hold_now = next((k for k, hd in enumerate(holds) if hd.t0 <= t <= hd.t1), None)
        blink = (i // max(1, int(fps * 0.35))) % 2 == 0

        # --- mirino: bordo scuro + griglia dei terzi + reticolo centrale ---
        ov = bgr.copy()
        cv2.rectangle(ov, (0, 0), (w, m), (0, 0, 0), -1)
        cv2.rectangle(ov, (0, h - m), (w, h), (0, 0, 0), -1)
        cv2.rectangle(ov, (0, 0), (m, h), (0, 0, 0), -1)
        cv2.rectangle(ov, (w - m, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, bgr, 0.55, 0, bgr)
        for gx in (w // 3, 2 * w // 3):
            cv2.line(bgr, (gx, m), (gx, h - m), (255, 255, 255), 1, cv2.LINE_AA)
        for gy in (m + (h - 2 * m) // 3, m + 2 * (h - 2 * m) // 3):
            cv2.line(bgr, (m, gy), (w - m, gy), (255, 255, 255), 1, cv2.LINE_AA)
        for sx in (m, w - m):
            for sy in (m, h - m):
                dx = 26 if sx == m else -26
                dy = 26 if sy == m else -26
                cv2.line(bgr, (sx, sy), (sx + dx, sy), (255, 255, 255), 2)
                cv2.line(bgr, (sx, sy), (sx, sy + dy), (255, 255, 255), 2)
        cv2.drawMarker(bgr, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 22, 1)

        # --- target del fuoco: bbox del fermo, o della presa, o del corpo ---
        tgt = None
        if hold_now is not None:
            tgt = holds[hold_now].bbox
        elif cts:
            c = next((c for c in cts if c.t0 <= t <= c.t1), None)
            if c:
                tgt = (c.px[0] - 0.13, c.px[1] - 0.13, c.px[0] + 0.13, c.px[1] + 0.13)
        if tgt is None and pf is not None and pf.lm is not None:
            tgt = _torso_bbox(pf.lm)
        if tgt is not None:
            tcx = (tgt[0] + tgt[2]) / 2 * w
            tcy = (tgt[1] + tgt[3]) / 2 * h
            thw = max(40, (tgt[2] - tgt[0]) / 2 * w)
            thh = max(40, (tgt[3] - tgt[1]) / 2 * h)
            for j, v in enumerate((tcx, tcy, thw, thh)):        # inseguimento morbido
                fx[j] += (v - fx[j]) * 0.35

        # --- griglia punti AF (verde quelli sul soggetto / target) ---
        for r in range(af_rows):
            for c in range(af_cols):
                px = int(m + (w - 2 * m) * (c + 0.5) / af_cols)
                py = int(m + (h - 2 * m) * (r + 0.5) / af_rows)
                inside = (fx[0] - fx[2] < px < fx[0] + fx[2] and
                          fx[1] - fx[3] < py < fx[1] + fx[3])
                col = GREEN if (inside and (hold_now is not None or blink)) else (150, 150, 150)
                s = 7 if inside else 4
                cv2.rectangle(bgr, (px - s, py - s), (px + s, py + s), col,
                              2 if inside else 1)

        # --- scheletro tenue + contatti ---
        if pf is not None and pf.lm is not None:
            L = pf.lm
            for a, b in _SKELETON:
                pa, pb = L[IDX[a]], L[IDX[b]]
                if pa[2] > 0.3 and pb[2] > 0.3:
                    cv2.line(bgr, (int(pa[0] * w), int(pa[1] * h)),
                             (int(pb[0] * w), int(pb[1] * h)), DIM, 1, cv2.LINE_AA)
            for part, names in GRIP_PARTS.items():
                xy = _part_xy(L, names, 0.3)
                if xy:
                    on = px_pole is not None and abs(xy[0] - px_pole) < 0.09
                    cv2.circle(bgr, (int(xy[0] * w), int(xy[1] * h)), 8,
                               GREEN if on else AMBER, -1 if on else 2)

        # --- palo ---
        if px_pole is not None:
            xp = int(px_pole * w)
            cv2.line(bgr, (xp, m), (xp, h - m), (0, 170, 255), 1, cv2.LINE_AA)
            cv2.putText(bgr, "POLE", (xp + 8, h // 2), FT, 0.5, (0, 170, 255), 1)

        # --- box di fuoco (staffe che scattano) ---
        locked = hold_now is not None
        col = GREEN if locked else AMBER
        bracket(bgr, fx[0], fx[1], fx[2], fx[3], col, 3 if locked else 2,
                int(min(fx[2], fx[3]) * 0.4))
        if locked and blink:
            cv2.rectangle(bgr, (int(fx[0] - fx[2]), int(fx[1] - fx[3])),
                          (int(fx[0] + fx[2]), int(fx[1] + fx[3])), GREEN, 1)

        # --- eventi scenici: inversione / estensione massima ---
        active_ev = [e for e in events if -0.2 <= t - e.t <= 0.55]
        for e in active_ev:
            d = t - e.t
            ec = EV_COL.get(e.kind, (255, 255, 255))
            ex, ey = int(e.xy[0] * w), int(e.xy[1] * h)
            rad = int(16 + max(0.0, d) * 240)                   # anello che si espande e sfuma
            cv2.circle(bgr, (ex, ey), rad, ec, 2, cv2.LINE_AA)
            cv2.drawMarker(bgr, (ex, ey), ec, cv2.MARKER_TILTED_CROSS, 22, 2)
        if active_ev:                                           # un solo tag per volta, in alto
            e = min(active_ev, key=lambda x: abs(t - x.t))
            ec = EV_COL.get(e.kind, (255, 255, 255))
            tag = (e.label or e.kind.upper()) + (f"  {e.value:.0f}°" if e.value else "")
            cv2.putText(bgr, tag, (m + 12, m + 44), FT, 0.7, ec, 2, cv2.LINE_AA)

        # --- HUD ---
        tc = f"{int(t // 60):02d}:{int(t % 60):02d}:{int((t * fps) % fps):02d}"
        if blink:
            cv2.circle(bgr, (m + 14, m + 16), 7, (60, 60, 255), -1)
        cv2.putText(bgr, f"REC {tc}", (m + 30, m + 22), FT, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr, "AF-C" if not locked else "AF LOCK", (w - m - 130, m + 22),
                    FT, 0.6, col, 2)
        lab = labels.get(hold_now, {}) if hold_now is not None else {}
        part_txt = PART_IT.get(holds[hold_now].focus_part, "?") if hold_now is not None else "--"
        grip_txt = lab.get("grip") or part_txt
        move_txt = lab.get("move") or ("POSE" if pf and pf.lm is not None else "no soggetto")
        cv2.putText(bgr, f"FOCUS: {grip_txt}", (m + 10, h - m - 14), FT, 0.6, col, 2)
        cv2.putText(bgr, move_txt.upper(), (w // 2 - 70, h - m - 14), FT, 0.55, (255, 255, 255), 1)
        if locked and blink:
            cv2.putText(bgr, "[ FOCUS LOCK ]", (w // 2 - 95, m + 46), FT, 0.6, GREEN, 2)

        # --- timeline dei fermi ---
        y = h - m + 8
        cv2.line(bgr, (m, y), (w - m, y), (90, 90, 90), 2)
        for hd in holds:
            x0 = int(m + hd.t0 / dur * (w - 2 * m))
            x1 = int(m + hd.t1 / dur * (w - 2 * m))
            cv2.line(bgr, (x0, y), (x1, y), GREEN, 6)
        for e in events:
            ex = int(m + e.t / dur * (w - 2 * m))
            cv2.drawMarker(bgr, (ex, y), EV_COL.get(e.kind, (255, 255, 255)),
                           cv2.MARKER_DIAMOND, 12, 2)
        cv2.drawMarker(bgr, (int(m + t / dur * (w - 2 * m)), y), (255, 255, 255),
                       cv2.MARKER_TRIANGLE_DOWN, 12, 2)

        vw.write(bgr)
    cap.release()
    vw.release()

    if transcode:
        import shutil
        import subprocess
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-preset", "veryfast",
                            str(out_path), "-loglevel", "error"], check=False)
            raw.unlink(missing_ok=True)
        else:
            raw.replace(out_path)
    return Path(out_path)


def save_frame(path: Path | str, t: float, out_png: Path | str) -> Path:
    """Estrae il fotogramma al secondo `t` come PNG."""
    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Nessun fotogramma a t={t}s in {path}")
    cv2.imwrite(str(out_png), bgr)
    return Path(out_png)


def capabilities() -> dict:
    return {"grip_parts": list(GRIP_PARTS), "model": _POSE_TASK.name,
            "model_present": _POSE_TASK.is_file()}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Analisi pose pole: contatti, fermi, anteprima.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("pose_preview.mp4"))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--people", type=int, default=1)
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=0.0)
    a = ap.parse_args()
    t1 = a.t1 or None

    frames = analyze_video(a.video, fps_sample=a.fps, max_people=a.people,
                           t_start=a.t0, t_end=t1)
    seen = sum(1 for f in frames if f.lm is not None)
    px, src = pole_x_auto(a.video, frames)
    cts = contacts(frames, px)
    holds = detect_holds(frames, cts)
    events = detect_events(frames)

    print(f"frame campionati: {len(frames)}  (persona rilevata in {seen})")
    print(f"palo: x={px:.3f}  ({src})" if px is not None else "palo: non trovato")
    print(f"contatti ({len(cts)}):")
    for c in cts:
        print(f"  {PART_IT.get(c.part, c.part):8s} {c.t0:5.2f}-{c.t1:5.2f}s  ({c.dur:.2f}s)")
    print(f"fermi ({len(holds)}):")
    for hd in holds:
        print(f"  {hd.t0:5.2f}-{hd.t1:5.2f}s  focus@{hd.focus_t:.2f}s  parte={PART_IT.get(hd.focus_part, hd.focus_part or '?')}")
    print(f"eventi ({len(events)}):")
    for e in events:
        print(f"  {e.t:5.2f}s  {e.kind:9s} {e.part:12s} {e.value:6.1f}  {e.label}")
    debug_video(a.video, a.out, frames, holds, cts, px, events=events, t_start=a.t0, t_end=t1)
    print(f"anteprima -> {a.out}")
