"""Grafica generata: template, validazione AssetSpec, sanitizzazione SVG."""
import pytest
from PIL import Image

from autoedit import assets
from autoedit.assets import (
    TEMPLATES, AssetSpec, asset_schema_text, capabilities, render_asset,
    sanitize_svg, text_mask_png, validate_asset_spec,
)


@pytest.mark.parametrize("name", TEMPLATES)
def test_every_template_renders_rgba_png(name, tmp_path):
    spec = AssetSpec(kind="template", template=name, params={"text": "TEST", "price": "9€"},
                     width=640, height=480, asset_id=name)
    out = render_asset(spec, tmp_path)
    assert out.is_file()
    im = Image.open(out)
    assert im.mode == "RGBA" and im.size == (640, 480)


def test_text_mask_is_black_transparent_with_white_text(tmp_path):
    out = text_mask_png("MOTO", 400, 400, tmp_path / "m.png")
    im = Image.open(out).convert("RGBA")
    alphas = im.getchannel("A").getextrema()
    assert alphas[0] == 0 and alphas[1] == 255       # trasparente fuori, pieno sul testo


def test_validate_asset_spec_template_ok():
    s = validate_asset_spec({"kind": "template", "template": "badge",
                             "params": {"text": "NEW"}, "width": 600, "height": 600,
                             "asset_id": "hey there!"})
    assert s.kind == "template" and s.template == "badge"
    assert s.asset_id == "heythere"                  # slug ripulito


def test_validate_asset_spec_unknown_template_falls_back():
    s = validate_asset_spec({"kind": "template", "template": "nope"})
    assert s.template == "title_card"


def test_validate_asset_spec_clamps_dimensions():
    s = validate_asset_spec({"kind": "template", "width": 999999, "height": 1})
    assert 64 <= s.width <= 4096 and 64 <= s.height <= 4096


def test_validate_asset_spec_rejects_bad_kinds():
    with pytest.raises(ValueError):
        validate_asset_spec({"kind": "text_mask"})              # manca text
    with pytest.raises(ValueError):
        validate_asset_spec({"kind": "svg", "svg": "not svg"})  # niente <svg>


def test_sanitize_svg_strips_script_and_external_refs():
    dirty = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
             '<script>alert(1)</script>'
             '<image href="http://evil/x.png"/>'
             '<rect width="10" height="10" fill="#f05"/></svg>')
    clean = sanitize_svg(dirty)
    assert "<script" not in clean.lower()
    assert "http://evil" not in clean
    assert "rect" in clean


def test_sanitize_svg_rejects_non_svg_root():
    with pytest.raises(ValueError):
        sanitize_svg("<div>nope</div>")


def test_schema_and_capabilities_shape():
    txt = asset_schema_text()
    assert "kind" in txt and all(t in txt for t in TEMPLATES)
    cap = capabilities()
    assert set(cap["kinds"]) == {"template", "svg", "text_mask"}
    assert cap["templates"] == list(TEMPLATES)


def test_render_asset_svg_requires_a_renderer(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "available_renderers", lambda: [])
    spec = AssetSpec(kind="svg",
                     svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 4">'
                         '<rect width="4" height="4"/></svg>')
    with pytest.raises(RuntimeError):
        render_asset(spec, tmp_path)
