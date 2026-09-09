"""Estrazione palette dai media (contesto visivo per l'LLM)."""
from PIL import Image

from autoedit.palette import describe, dominant_colors


def _solid(tmp_path, name, rgb):
    p = tmp_path / name
    Image.new("RGB", (200, 200), rgb).save(p)
    return p


def test_dominant_colors_recovers_solid_hues(tmp_path):
    imgs = [_solid(tmp_path, "a.png", (110, 31, 214)),   # viola
            _solid(tmp_path, "b.png", (67, 255, 19))]     # verde
    cols = dominant_colors(imgs, k=4)
    assert cols and all(c.startswith("#") and len(c) == 7 for c in cols)
    # i due colori pieni devono comparire (vicini)
    def near(hx, rgb):
        v = tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))
        return sum((a - b) ** 2 for a, b in zip(v, rgb)) < 2500
    assert any(near(c, (110, 31, 214)) for c in cols)
    assert any(near(c, (67, 255, 19)) for c in cols)


def test_describe_adds_colour_names(tmp_path):
    txt = describe([_solid(tmp_path, "p.png", (110, 31, 214))], k=2)
    assert "#" in txt and "viola" in txt


def test_empty_or_bad_paths_return_empty():
    assert dominant_colors([]) == []
    assert dominant_colors(["/nope/x.png"]) == []
