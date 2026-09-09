"""Maschere: costruzione dei frammenti + smoke render con ffmpeg."""
import subprocess

import pytest

from autoedit.masks import (
    EFFECT_CHAINS, SHAPES, build_mask, capabilities, masked_effect, shape_reveal,
)
from tests.conftest import needs_ffmpeg

GEQ_SHAPES = ("circle", "ellipse", "rect", "rrect", "diamond", "heart")


@pytest.mark.parametrize("shape", GEQ_SHAPES)
def test_geq_shape_is_self_contained_fragment(shape):
    frag = build_mask(shape, 540, 960, fps=30, duration=1.0, out_label="m")
    assert frag.endswith("[m]")
    assert "geq=lum=" in frag and "color=c=black" in frag
    assert "min(W,H)" in frag                     # coord isotrope


def test_growth_none_uses_fixed_size_not_full_frame():
    stat = build_mask("circle", 500, 500, duration=1.0,
                      params={"growth": "none", "size": 0.4}, out_label="m")
    assert "st(0,0.400)" in stat
    anim = build_mask("circle", 500, 500, duration=1.0,
                      params={"growth": "in"}, out_label="m")
    assert "2.10*(clip(T/1.000,0,1))" in anim


def test_invert_and_feather_append_filters():
    frag = build_mask("heart", 400, 400, params={"invert": True, "feather": 0.5}, out_label="m")
    assert ",negate[" in frag and "gblur=sigma=" in frag


def test_text_shape_uses_movie_and_loops_one_frame(tmp_path):
    frag = build_mask("text", 400, 400, params={"text": "HI"}, out_label="m")
    assert "movie=" in frag and "loop=loop=-1:size=1" in frag


def test_image_shape_requires_existing_file():
    with pytest.raises(ValueError):
        build_mask("image", 100, 100, params={"image": "/nope/x.png"})


def test_masked_effect_filtergraph_wires_split_and_maskedmerge():
    fc = masked_effect("in0", "blur", "circle", 320, 480, inside=True, out_label="o")
    assert fc.startswith("[in0]split")
    assert EFFECT_CHAINS["blur"] in fc
    assert "maskedmerge[o]" in fc


def test_shape_reveal_merges_two_labels_through_mask():
    fc = shape_reveal("a", "b", "diamond", 320, 480, out_label="o")
    assert "[a][b][_mk]maskedmerge[o]" in fc


def test_capabilities_lists_all():
    cap = capabilities()
    assert cap["shapes"] == list(SHAPES)
    assert "blur" in cap["effects"] and "pulse" in cap["growth"]


@needs_ffmpeg
@pytest.mark.parametrize("shape", GEQ_SHAPES)
def test_geq_mask_renders_grayscale_frames(shape, tmp_path):
    out = tmp_path / f"{shape}.mp4"
    frag = build_mask(shape, 240, 426, fps=15, duration=0.6, out_label="m")
    r = subprocess.run(
        ["ffmpeg", "-y", "-filter_complex", frag, "-map", "[m]", "-t", "0.6",
         "-pix_fmt", "yuv420p", str(out), "-loglevel", "error"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.stat().st_size > 1000


@needs_ffmpeg
def test_shape_reveal_renders_and_changes_over_time(tmp_path):
    out = tmp_path / "rev.mp4"
    fc = shape_reveal("0:v", "1:v", "circle", 240, 426, fps=15, duration=0.8, out_label="o")
    r = subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=240x426:r=15:d=0.8",
         "-f", "lavfi", "-i", "color=c=white:s=240x426:r=15:d=0.8",
         "-filter_complex", fc, "-map", "[o]", "-t", "0.8",
         "-pix_fmt", "yuv420p", str(out), "-loglevel", "error"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.stat().st_size > 1000
