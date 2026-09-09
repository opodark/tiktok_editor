"""Pipeline end-to-end (richiede ffmpeg): crop globale + overlay generati."""
import subprocess

from autoedit.assets import AssetSpec, render_asset
from autoedit.pipeline import ASPECTS, RenderConfig, run_pipeline
from tests.conftest import needs_ffmpeg


def _dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


@needs_ffmpeg
def test_run_pipeline_9x16_with_crop_and_overlay(tmp_path, media_dir, audio_file):
    badge = render_asset(
        AssetSpec(kind="template", template="badge", params={"text": "OK"},
                  width=400, height=400, asset_id="ov"),
        tmp_path)
    out = tmp_path / "reel.mp4"
    cfg = RenderConfig(
        media_dir=media_dir, audio=audio_file, out=out,
        aspect="9:16", edit_style="clean", seed=1, max_duration=2.0,
        crop_zoom=1.2, crop_y=0.2,
        overlay_specs=({"png": str(badge), "pos": "tr", "scale": 0.25},),
    )
    run_pipeline(cfg)
    assert out.is_file() and out.stat().st_size > 5000
    assert _dims(out) == ASPECTS["9:16"]


@needs_ffmpeg
def test_run_pipeline_default_crop_matches_old_behaviour(tmp_path, media_dir, audio_file):
    out = tmp_path / "reel.mp4"
    cfg = RenderConfig(media_dir=media_dir, audio=audio_file, out=out,
                       aspect="1:1", edit_style="clean", seed=2, max_duration=1.5)
    run_pipeline(cfg)
    assert _dims(out) == ASPECTS["1:1"]
