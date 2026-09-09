"""LLMConfig + validazione override del brief (nessuna rete)."""
import json

from autoedit.brief import LLMConfig, load_llm_config, save_llm_config, validate_overrides


def test_llmconfig_defaults_point_at_local_ollama():
    c = LLMConfig()
    assert c.provider == "openai"
    assert c.base_url == "http://localhost:11434/v1"
    assert hasattr(c, "asset_model")


def test_config_roundtrip(tmp_path, monkeypatch):
    import autoedit.brief as brief
    p = tmp_path / "llm_config.json"
    monkeypatch.setattr(brief, "_CONFIG_PATH", p)
    save_llm_config(LLMConfig(model="m1", asset_model="coder1", base_url="http://x/v1"))
    got = load_llm_config()
    assert got.model == "m1" and got.asset_model == "coder1"
    assert "asset_model" in json.loads(p.read_text())


def test_config_ignores_unknown_keys(tmp_path, monkeypatch):
    import autoedit.brief as brief
    p = tmp_path / "llm_config.json"
    p.write_text('{"model": "ok", "bogus": 1}')
    monkeypatch.setattr(brief, "_CONFIG_PATH", p)
    assert load_llm_config().model == "ok"


def test_validate_overrides_keeps_known_clamps_ranges():
    ov = validate_overrides({
        "edit_style": "hype", "style": "neon", "motion": "bounce",
        "variety": 5.0, "grain": -1, "cuts_per_beat": 2, "xfade_ms": 9999,
        "vignette": True, "unknown_key": "x", "impact_effects": ["flash", "bogus"],
    })
    assert ov["edit_style"] == "hype" and ov["style"] == "neon"
    assert ov["variety"] == 1.0 and ov["grain"] == 0.0
    assert ov["cuts_per_beat"] == 2.0 and ov["xfade_ms"] == 400
    assert ov["vignette"] is True
    assert ov["impact_effects"] == ["flash"]
    assert "unknown_key" not in ov


def test_validate_overrides_drops_invalid_enums():
    ov = validate_overrides({"edit_style": "nope", "aspect": "3:2", "cuts_per_beat": 1.7})
    assert ov == {}
