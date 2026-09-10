"""pose.py: logica di contatti / fermi (senza MediaPipe ne' video)."""
import numpy as np
import pytest

from autoedit.pose import (
    GRIP_PARTS, IDX, Contact, PoseFrame, _motion, _part_xy, capabilities, contacts, detect_holds,
)


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
    for i in range(20):
        shift = 0.0 if i < 8 or i > 12 else (i - 8) * 0.05   # movimento a meta'
        b = {k: (x + shift, y) for k, (x, y) in body.items()}
        frames.append(PoseFrame(t=i * 0.15, lm=_lm(**b)))
    holds = detect_holds(frames, [], still=0.012, min_hold=0.3)
    assert len(holds) == 2
    assert holds[0].t1 <= holds[1].t0


def test_hold_focus_is_most_recent_contact():
    frames = [PoseFrame(t=i * 0.1, lm=np.zeros((33, 3), dtype=np.float32) + [0.5, 0.5, 1.0])
              for i in range(20)]
    cts = [Contact("left_hand", 0.0, 2.0, (0.5, 0.4)),
           Contact("right_foot", 0.7, 2.0, (0.5, 0.9))]
    holds = detect_holds(frames, cts, still=0.05, min_hold=0.3)
    assert holds and holds[0].focus_part == "right_foot"    # iniziato dopo


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
