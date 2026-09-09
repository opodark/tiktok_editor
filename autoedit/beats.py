"""Beat detection reale su un file audio, via librosa.

Oltre a tempo e griglia di beat stimiamo:
- i *beat forti*  (accento marcato) -> transizioni glitch/flash;
- i *downbeat*     (primo movimento della battuta, assunto 4/4) -> taglio
  piu' deciso, con un "punch" di zoom piu' marcato;
- i *transienti*   reali (picchi di onset) a cui agganciare i tagli
  quando si sottodivide il beat (cuts-per-beat >= 2), cosi' gli ottavi
  cadono su qualcosa che suona davvero e non su una griglia teorica.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

_HOP = 512


@dataclass
class BeatInfo:
    tempo: float
    beat_times: np.ndarray       # istanti (s) di ogni beat rilevato
    strong_times: np.ndarray     # beat con accento forte
    downbeat_times: np.ndarray   # beat "1" di ogni battuta (stima 4/4)
    onset_times: np.ndarray      # transienti reali del brano
    duration: float
    sr: int = 22050
    onset_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hop_length: int = _HOP

    @property
    def beat_period(self) -> float:
        if len(self.beat_times) >= 2:
            return float(np.median(np.diff(self.beat_times)))
        return 60.0 / max(1.0, self.tempo)


def _energy_at(beat_info: "BeatInfo", t: float, half: float = 0.12) -> float:
    """Onset strength media in una finestra di +-half secondi attorno a t."""
    oe = beat_info.onset_env
    if len(oe) == 0:
        return 0.0
    f0, f1 = librosa.time_to_frames(
        [max(0.0, t - half), t + half], sr=beat_info.sr, hop_length=beat_info.hop_length
    )
    f0 = max(0, int(f0))
    f1 = min(len(oe), int(f1))
    return float(np.mean(oe[f0:f1])) if f1 > f0 else 0.0


def _estimate_downbeats(beat_times: np.ndarray, beat_strengths: np.ndarray,
                         meter: int = 4) -> np.ndarray:
    """Sceglie la fase (0..meter-1) con l'accento medio piu' alto:
    quei beat sono i downbeat della battuta."""
    if len(beat_times) < meter:
        return beat_times[:1]
    best_phase, best_score = 0, -1.0
    for phase in range(meter):
        sel = beat_strengths[phase::meter]
        score = float(np.mean(sel)) if len(sel) else -1.0
        if score > best_score:
            best_phase, best_score = phase, score
    return beat_times[best_phase::meter]


def detect_beats(audio_path: Path, tightness: int = 100) -> BeatInfo:
    """Rileva tempo (BPM), beat, beat forti, downbeat e transienti."""
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = float(len(y) / sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=_HOP, tightness=tightness, units="frames"
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=_HOP)

    if len(beat_times) == 0:
        # nessun beat rilevabile -> griglia sintetica a 120 BPM
        step = 60.0 / 120.0
        beat_times = np.arange(0, duration, step)
        tempo = 120.0

    frame_idx = np.clip(
        librosa.time_to_frames(beat_times, sr=sr, hop_length=_HOP), 0, len(onset_env) - 1
    )
    strengths = onset_env[frame_idx]
    threshold = np.median(strengths) + 0.15 * np.std(strengths)
    strong_times = beat_times[strengths >= threshold]
    if len(strong_times) == 0:
        strong_times = beat_times[::4]

    downbeat_times = _estimate_downbeats(beat_times, strengths)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=_HOP, backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=_HOP)

    tempo_val = float(tempo) if np.isscalar(tempo) else float(np.atleast_1d(tempo)[0])
    return BeatInfo(
        tempo=tempo_val,
        beat_times=np.asarray(beat_times, float),
        strong_times=np.asarray(strong_times, float),
        downbeat_times=np.asarray(downbeat_times, float),
        onset_times=np.asarray(onset_times, float),
        duration=duration,
        sr=int(sr),
        onset_env=np.asarray(onset_env, float),
        hop_length=_HOP,
    )


def _snap_to_onsets(cuts: np.ndarray, protected: np.ndarray, beat_info: BeatInfo,
                     frac: float = 0.14) -> np.ndarray:
    """Sposta ogni taglio "intermedio" sul transiente reale piu' vicino,
    se cade entro `frac` del periodo di beat. I tagli che coincidono con
    un beat vero (`protected`) restano dove sono."""
    onsets = beat_info.onset_times
    if len(onsets) == 0:
        return cuts
    win = beat_info.beat_period * frac
    prot = set(np.round(protected, 4).tolist())
    out = []
    for t in cuts:
        if round(float(t), 4) in prot:
            out.append(float(t))
            continue
        j = int(np.argmin(np.abs(onsets - t)))
        out.append(float(onsets[j]) if abs(onsets[j] - t) <= win else float(t))
    return np.array(out)


def get_cut_times(beat_info: BeatInfo, cuts_per_beat: float, *,
                   strong_only: bool = False, dynamic: bool = False,
                   snap: bool = True) -> np.ndarray:
    """Deriva i punti di taglio dal beat grid.

    cuts_per_beat = 1   -> un taglio per beat
    cuts_per_beat = 0.5 -> un taglio ogni 2 beat (piu' lento)
    cuts_per_beat = 2   -> due tagli per beat, cioe' sugli ottavi

    strong_only  taglia solo sui beat con accento marcato.
    dynamic      raddoppia i tagli nei tratti a energia alta e li dimezza
                 nei tratti calmi, attorno al valore di cuts_per_beat.
    snap         aggancia i tagli sottodivisi ai transienti reali.
    """
    base = np.asarray(beat_info.strong_times if strong_only else beat_info.beat_times, float)
    if len(base) < 2:
        return base

    cpb = float(cuts_per_beat)
    if cpb < 1.0 and not dynamic:
        step = max(1, int(round(1.0 / cpb)))
        base = base[::step]
        cpb = 1.0

    if dynamic:
        beat_energies = np.array([_energy_at(beat_info, bt) for bt in beat_info.beat_times])
        hi = float(np.percentile(beat_energies, 70)) if len(beat_energies) else 0.0
        lo = float(np.percentile(beat_energies, 35)) if len(beat_energies) else 0.0

    times: list[float] = []
    for i in range(len(base) - 1):
        a, b = float(base[i]), float(base[i + 1])
        local = cpb
        if dynamic:
            e = _energy_at(beat_info, a)
            if e >= hi:
                local = min(4.0, cpb * 2)
            elif e <= lo:
                local = max(0.5, cpb / 2)
        n = max(1, int(round(local)))
        for f in range(n):
            times.append(a + (b - a) * f / n)
    times.append(float(base[-1]))

    cuts = np.array(sorted(times))
    if snap:
        cuts = np.array(sorted(_snap_to_onsets(cuts, base, beat_info)))

    # dedup + gap minimo, cosi' lo snapping non crea tagli sovrapposti
    out = [float(cuts[0])]
    for t in cuts[1:]:
        if float(t) - out[-1] >= 0.12:
            out.append(float(t))
    return np.array(out)


def is_accented(t: float, beat_info: BeatInfo, tol: float = 0.08) -> bool:
    """True se l'istante t cade (entro tol secondi) su un beat accentato."""
    if len(beat_info.strong_times) == 0:
        return False
    return bool(np.any(np.abs(beat_info.strong_times - t) <= tol))


def is_downbeat(t: float, beat_info: BeatInfo, tol: float = 0.08) -> bool:
    """True se l'istante t cade sul primo movimento di una battuta."""
    if len(beat_info.downbeat_times) == 0:
        return False
    return bool(np.any(np.abs(beat_info.downbeat_times - t) <= tol))
