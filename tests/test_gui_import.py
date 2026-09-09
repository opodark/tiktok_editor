"""La GUI si costruisce senza errori (prende i bug di wiring di Gradio)."""
import importlib
import warnings


def test_gui_module_builds_blocks():
    warnings.simplefilter("ignore")
    gui = importlib.import_module("gui")
    assert gui.demo is not None                 # gr.Blocks costruito = wiring ok
    assert callable(gui._crop_preview)
    assert "9:16" in gui.ASPECT_LABELS.values()


def test_crop_preview_helper_produces_two_images(tmp_path):
    from PIL import Image
    warnings.simplefilter("ignore")
    gui = importlib.import_module("gui")
    src = tmp_path / "p.jpg"
    Image.new("RGB", (1600, 900), "#3355ff").save(src)
    res, orig = gui._crop_preview(src, "9:16", 1.3, 0.5, 0.3)
    assert res.size[0] < res.size[1]            # verticale
    assert orig.size[0] > 0
