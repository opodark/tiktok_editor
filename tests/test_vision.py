"""vision.py: costruzione richiesta + parsing risposte (nessuna rete)."""
import pytest

from autoedit import vision
from autoedit.brief import LLMConfig


def test_data_uri_normalises_jpg(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff\xd8\xff")
    uri = vision._data_uri(p)
    assert uri.startswith("data:image/jpeg;base64,")


def test_vision_model_falls_back_to_default():
    assert vision._vision_model(LLMConfig()) == vision.DEFAULT_VISION_MODEL
    assert vision._vision_model(LLMConfig(vision_model="minicpm-v")) == "minicpm-v"


def test_grip_part_parses_model_json(monkeypatch, tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(vision, "ask_image",
                        lambda *a, **k: 'ecco: {"parts": ["mano destra"], '
                                        '"load_bearing": "mano destra", "on_pole": true}')
    out = vision.grip_part(LLMConfig(), img)
    assert out["parts"] == ["mano destra"] and out["on_pole"] is True


def test_grip_part_survives_garbage(monkeypatch, tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(vision, "ask_image", lambda *a, **k: "non lo so, scusa")
    out = vision.grip_part(LLMConfig(), img)
    assert out == {"parts": [], "load_bearing": "", "on_pole": False}


def test_name_pose_clamps_confidence(monkeypatch, tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(vision, "ask_image",
                        lambda *a, **k: '{"move": "Gemini", "held": true, "confidence": 5}')
    out = vision.name_pose(LLMConfig(), img, ["Gemini", "Superman"])
    assert out["move"] == "Gemini" and out["confidence"] == 1.0
