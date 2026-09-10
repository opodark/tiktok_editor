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


# --------------------------------------------------------------------------
def analyze_video(path: Path | str, fps_sample: float = 8.0, max_people: int = 1) -> list[PoseFrame]:
    """Campiona il video a ~`fps_sample` e ritorna i keypoint per frame."""
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

    frames: list[PoseFrame] = []
    with PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % step == 0:
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
    for f in np.linspace(total * 0.1, total * 0.9, probe_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        edges = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 60, 180)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                                minLineLength=int(h * 0.45), maxLineGap=25)
        if lines is None:
            continue
        cand = []
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang > 78:                                 # quasi verticale
                cand.append((x1 + x2) / 2.0 / w)
        if cand:
            xs.append(float(np.median(cand)))
    cap.release()
    return float(np.median(xs)) if xs else None


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


def detect_holds(frames: list[PoseFrame], cts: list[Contact],
                 still: float = 0.012, min_hold: float = 0.45) -> list[Hold]:
    """Tratti fermi (movimento medio dei keypoint < `still`) lunghi almeno
    `min_hold` s. Il FOCUS e' il contatto iniziato piu' di recente."""
    holds: list[Hold] = []
    run_start = None
    lowest = (1e9, 0.0)
    for i in range(1, len(frames)):
        mv = _motion(frames[i - 1].lm, frames[i].lm)
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
    a = ap.parse_args()

    frames = analyze_video(a.video, fps_sample=a.fps, max_people=a.people)
    seen = sum(1 for f in frames if f.lm is not None)
    px = pole_x(a.video)
    cts = contacts(frames, px)
    holds = detect_holds(frames, cts)

    print(f"frame campionati: {len(frames)}  (persona rilevata in {seen})")
    print(f"palo: x={px:.3f}" if px is not None else "palo: non trovato")
    print(f"contatti ({len(cts)}):")
    for c in cts:
        print(f"  {PART_IT.get(c.part, c.part):8s} {c.t0:5.2f}-{c.t1:5.2f}s  ({c.dur:.2f}s)")
    print(f"fermi ({len(holds)}):")
    for hd in holds:
        print(f"  {hd.t0:5.2f}-{hd.t1:5.2f}s  focus@{hd.focus_t:.2f}s  parte={PART_IT.get(hd.focus_part, hd.focus_part or '?')}")
    annotate(a.video, a.out, frames, holds, px)
    print(f"anteprima -> {a.out}")
