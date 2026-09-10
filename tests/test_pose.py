"""pose.py: logica di contatti / fermi (senza MediaPipe ne' video)."""
import numpy as np
import pytest

from autoedit.pose import (
    GRIP_PARTS, IDX, Contact, PoseFrame, _ang, _bbox_visible, _motion, _part_xy, _torso_bbox,
    body_metrics, capabilities, contacts, detect_events, detect_holds,
)


def _full(shift=0.0, flip=False):
    """Corpo completo in piedi (flip=True -> a testa in giu')."""
    base = dict(l_shoulder=(0.45, 0.30), r_shoulder=(0.55, 0.30),
                l_elbow=(0.40, 0.45), r_elbow=(0.60, 0.45),
                l_wrist=(0.38, 0.60), r_wrist=(0.62, 0.60),
                l_hip=(0.46, 0.55), r_hip=(0.54, 0.55),
                l_knee=(0.46, 0.75), r_knee=(0.54, 0.75),
                l_ankle=(0.46, 0.95), r_ankle=(0.54, 0.95))
    if flip:
        base = {k: (x, 1.0 - y) for k, (x, y) in base.items()}
    base = {k: (x + shift, y) for k, (x, y) in base.items()}
    return _lm(**base)


def _lm(**pos):
    """33 landmark; `pos` = {nome: (x, y)} visibili, gli altri invisibili."""
    a = np.zeros((33, 3), dtype=np.float32)
    for name, (x, y) in pos.items():
        a[IDX[name]] = (x, y, 1.0)
    return a


def test_part_xy_averages_visible_only():
    lm = _lm(l_wrist=(0.4, 0.5), l_index=(0.6, 0.5))   # l_thumb invisibile
    assert _part_xy(lm, GRIP_PARTS["left_hand"]) == (0.5, 0.5)
    assert _part_xy(_lm(), GRIP_PARTS["left_hand"]) is None


def test_motion_zero_when_still_one_when_missing():
    lm = _lm(l_shoulder=(0.4, 0.3), r_shoulder=(0.6, 0.3), l_hip=(0.42, 0.6),
             r_hip=(0.58, 0.6), l_knee=(0.42, 0.8), r_knee=(0.58, 0.8),
             l_ankle=(0.42, 0.95), r_ankle=(0.58, 0.95))
    assert _motion(lm, lm.copy()) == 0.0
    assert _motion(None, lm) == 1.0


def test_contacts_needs_pole_and_proximity():
    frames = [PoseFrame(t=i * 0.2, lm=_lm(l_wrist=(0.5, 0.4), l_index=(0.5, 0.4),
                                          l_thumb=(0.5, 0.4)))
              for i in range(6)]
    assert contacts(frames, None) == []                     # nessun palo
    cts = contacts(frames, px_pole=0.5, band=0.09)
    assert any(c.part == "left_hand" for c in cts)
    far = contacts(frames, px_pole=0.9, band=0.05)          # mano lontana dal palo
    assert far == []


def test_detect_holds_splits_on_motion():
    body = dict(l_shoulder=(0.4, 0.3), r_shoulder=(0.6, 0.3), l_hip=(0.42, 0.6),
                r_hip=(0.58, 0.6), l_knee=(0.42, 0.8), r_knee=(0.58, 0.8),
                l_ankle=(0.42, 0.95), r_ankle=(0.58, 0.95))
    frames = []
    for i in range(24):
        # posa A (0-8), transizione A->B (9-13), posa B ferma (14-23)
        shift = 0.0 if i <= 8 else (0.5 if i >= 14 else (i - 8) / 6 * 0.5)
        b = {k: (x + shift, y) for k, (x, y) in body.items()}
        frames.append(PoseFrame(t=i * 0.15, lm=_lm(**b)))
    holds = detect_holds(frames, [], still=0.015, min_hold=0.3)
    assert len(holds) == 2
    assert holds[0].t1 <= holds[1].t0


def test_hold_focus_is_most_recent_contact():
    frames = [PoseFrame(t=i * 0.1, lm=np.zeros((33, 3), dtype=np.float32) + [0.5, 0.5, 1.0])
              for i in range(20)]
    cts = [Contact("left_hand", 0.0, 2.0, (0.5, 0.4)),
           Contact("right_foot", 0.7, 2.0, (0.5, 0.9))]
    holds = detect_holds(frames, cts, still=0.05, min_hold=0.3)
    assert holds and holds[0].focus_part == "right_foot"    # iniziato dopo


def test_torso_bbox_is_centered_and_clamped():
    lm = _lm(l_shoulder=(0.35, 0.30), r_shoulder=(0.65, 0.30),
             l_hip=(0.40, 0.60), r_hip=(0.60, 0.60))
    x0, y0, x1, y1 = _torso_bbox(lm)
    assert x0 < 0.5 < x1 and y0 < 0.45 < y1
    assert (x1 - x0) / 2 <= 0.24 + 1e-6            # raggio clampato
    assert _torso_bbox(_lm()) is None             # niente busto visibile -> None come _bbox_visible


def test_bbox_visible_none_when_too_few_points():
    assert _bbox_visible(_lm(l_wrist=(0.5, 0.5))) is None
    b = _bbox_visible(_lm(l_wrist=(0.2, 0.3), r_wrist=(0.8, 0.3),
                          l_ankle=(0.3, 0.9), r_ankle=(0.7, 0.9)))
    assert b == pytest.approx((0.2, 0.3, 0.8, 0.9), abs=1e-5)


def test_angle_helper():
    assert _ang((0, 1, 1), (0, 0, 1), (1, 0, 1)) == pytest.approx(90, abs=0.5)
    assert _ang((-1, 0, 1), (0, 0, 1), (1, 0, 1)) == pytest.approx(180, abs=0.5)


def test_body_metrics_detects_inversion_and_extension():
    up = body_metrics(_full())
    assert up["inverted"] is False and abs(up["torso_tilt"]) < 10
    assert up["braccio sx"] == pytest.approx(180, abs=25)   # arti stesi nel manichino

    down = body_metrics(_full(flip=True))
    assert down["inverted"] is True
    assert abs(down["torso_tilt"]) > 150


def test_detect_events_finds_invert_run_and_extension_peak():
    frames = []
    for i in range(18):
        if 6 <= i <= 11:                       # 6 frame a testa in giu'
            lm = _full(flip=True)
        else:
            lm = _full()
            # a i==2 pieghi il braccio sx, poi lo stendi -> picco di estensione a i==3..4
            if i in (1, 2):
                lm[IDX["l_wrist"]] = (0.46, 0.44, 1.0)   # gomito piegato
        frames.append(PoseFrame(t=i * 0.15, lm=lm))
    ev = detect_events(frames)
    kinds = {e.kind for e in ev}
    assert "invert" in kinds
    inv = [e for e in ev if e.kind == "invert"][0]
    assert 0.8 < inv.t < 1.8                    # dentro la finestra capovolta


def test_capabilities():
    cap = capabilities()
    assert set(cap["grip_parts"]) == set(GRIP_PARTS)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mediapipe") is None,
    reason="mediapipe non installato")
def test_mediapipe_and_model_available():
    import mediapipe  # noqa: F401
    from autoedit.pose import _POSE_TASK
    assert _POSE_TASK.is_file()
