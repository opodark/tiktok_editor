"""crop_to_fill: inquadratura globale (zoom + ancora)."""
from autoedit.effects import crop_to_fill


def test_center_crop_is_default_and_unchanged():
    f = crop_to_fill(1920, 1080, 1080, 1920)
    assert "scale=1080:1920:force_original_aspect_ratio=increase" in f
    assert "x=(iw-ow)*0.5000" in f and "y=(ih-oh)*0.5000" in f


def test_anchor_moves_crop_window():
    top = crop_to_fill(1920, 1080, 1080, 1920, 1.0, 0.5, 0.0)
    assert "y=(ih-oh)*0.0000" in top
    right = crop_to_fill(1920, 1080, 1080, 1920, 1.0, 1.0, 0.5)
    assert "x=(iw-ow)*1.0000" in right


def test_zoom_enlarges_scale_target():
    f = crop_to_fill(1920, 1080, 1080, 1920, 2.0, 0.5, 0.5)
    assert "scale=2160:3840:force_original_aspect_ratio=increase" in f
    assert "crop=1080:1920" in f


def test_params_are_clamped():
    f = crop_to_fill(100, 100, 100, 100, zoom=0.1, ax=-3.0, ay=9.0)
    assert "scale=100:100" in f            # zoom < 1 -> 1
    assert "x=(iw-ow)*0.0000" in f and "y=(ih-oh)*1.0000" in f
