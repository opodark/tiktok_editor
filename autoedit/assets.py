"""Asset grafici generati: PNG RGBA pronti per l'overlay / la maschera.

Due strade, nessuna GPU, nessun modello di diffusione (e' roba vettoriale/testo):

- **template**  -> `render_template(name, params, ...)`: costruiamo noi il PNG con
  PIL da una manciata di parametri. Poco "slop", massimo controllo. Adatto a
  wordmark, lower-third, @handle, badge, title card, cartellino prezzo, CTA.
- **SVG grezzo** -> `render_svg(svg, ...)`: l'LLM (o l'utente) fornisce codice SVG,
  noi lo sanifichiamo e lo rasterizziamo col primo motore disponibile
  (`resvg` sul PATH, altrimenti `cairosvg`). Per quando c'e' voglia di
  sbizzarrirsi con forme libere.

`text_mask_png()` produce testo bianco su trasparente: si usa come MASCHERA in
ffmpeg (il video che scorre dentro le lettere).

I font inclusi (`assets_data/fonts/`) sono SIL OFL, quindi ok anche per uso
commerciale, e rendono il risultato identico su macOS e Windows.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = Path(__file__).resolve().parent / "assets_data" / "fonts"
_DISPLAY = _FONT_DIR / "Anton-Regular.ttf"        # titoli grossi, stile TikTok
_TEXT = _FONT_DIR / "Inter-Variable.ttf"          # testo pulito, @handle, sottotitoli

TEMPLATES = ("wordmark", "title_card", "lower_third", "handle", "badge", "price_tag", "cta")


# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
def _font(path: Path, size: int, weight: Optional[int] = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(path), size=size)
    if weight is not None:
        try:
            # font variabile: imposta l'asse "wght" e lascia gli altri (es. "opsz")
            # al default, altrimenti set_variation_by_axes assegna i valori in ordine
            # e finiremmo per scrivere il peso nell'asse sbagliato.
            axes = f.get_variation_axes()
            vals = []
            for ax in axes:
                nm = ax["name"]
                nm = nm.decode("ascii", "ignore") if isinstance(nm, bytes) else nm
                if nm.lower().startswith("weight"):
                    vals.append(max(ax["minimum"], min(ax["maximum"], weight)))
                else:
                    vals.append(ax["default"])
            if vals:
                f.set_variation_by_axes(vals)
        except Exception:  # noqa: BLE001  (font non variabile: ignora)
            pass
    return f


def display_font(size: int) -> ImageFont.FreeTypeFont:
    return _font(_DISPLAY, size)


def text_font(size: int, weight: int = 600) -> ImageFont.FreeTypeFont:
    return _font(_TEXT, size, weight)


# ---------------------------------------------------------------------------
# Utilita' di disegno
# ---------------------------------------------------------------------------
def _canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _hex(c: str, alpha: int = 255) -> tuple[int, int, int, int]:
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (r, g, b, alpha)


def _fit_font(font_path: Path, text: str, max_w: int, start: int,
              min_size: int = 12, weight: Optional[int] = None) -> ImageFont.FreeTypeFont:
    """Piu' grande dimensione che fa stare `text` in `max_w`."""
    size = start
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    while size > min_size:
        f = _font(font_path, size, weight)
        if probe.textlength(text, font=f) <= max_w:
            return f
        size -= 2
    return _font(font_path, min_size, weight)


def _text_with_shadow(d: ImageDraw.ImageDraw, xy, text, font, fill, shadow=(0, 0, 0, 150),
                       offset: int = 4, anchor: Optional[str] = None) -> None:
    x, y = xy
    d.text((x + offset, y + offset), text, font=font, fill=shadow, anchor=anchor)
    d.text((x, y), text, font=font, fill=fill, anchor=anchor)


def _rounded(d: ImageDraw.ImageDraw, box, radius, fill) -> None:
    d.rounded_rectangle(box, radius=radius, fill=fill)


# ---------------------------------------------------------------------------
# Template (PIL)
# ---------------------------------------------------------------------------
@dataclass
class AssetSpec:
    """Descrizione dichiarativa di un asset. La emette l'LLM, la validiamo qui."""
    kind: str = "template"                 # "template" | "svg" | "text_mask"
    template: str = "wordmark"             # se kind == "template"
    params: dict = field(default_factory=dict)
    svg: str = ""                          # se kind == "svg"
    text: str = ""                         # se kind == "text_mask"
    width: int = 1080
    height: int = 1080
    asset_id: str = "asset"


def _t_wordmark(p: dict, w: int, h: int) -> Image.Image:
    """Logo testuale: una o due parole, sfondo trasparente."""
    img, d = _canvas(w, h)
    text = (p.get("text") or "BRAND").upper()
    color = _hex(p.get("color", "#ffffff"))
    accent = p.get("accent")               # colora l'ultima parola
    f = _fit_font(_DISPLAY, text, int(w * 0.9), int(h * 0.6))
    if accent and " " in text:
        head, tail = text.rsplit(" ", 1)
        fw_head = d.textlength(head + " ", font=f)
        total = d.textlength(text, font=f)
        x0 = (w - total) / 2
        y = h / 2
        _text_with_shadow(d, (x0, y), head + " ", f, color, anchor="lm")
        _text_with_shadow(d, (x0 + fw_head, y), tail, f, _hex(accent), anchor="lm")
    else:
        _text_with_shadow(d, (w / 2, h / 2), text, f, color, anchor="mm")
    return img


def _t_title_card(p: dict, w: int, h: int) -> Image.Image:
    """Titolo grande su barra piena, centrato: buono come card d'apertura."""
    img, d = _canvas(w, h)
    text = (p.get("text") or "TITOLO").upper()
    bar = _hex(p.get("bg", "#ff0050"), int(255 * float(p.get("bg_opacity", 0.92))))
    fg = _hex(p.get("color", "#ffffff"))
    lines = _wrap_display(text, int(w * 0.82), int(h * 0.14))
    f = lines[1]
    lh = int(f.size * 1.15)
    block_h = lh * len(lines[0])
    pad = int(f.size * 0.55)
    top = (h - block_h) // 2 - pad
    _rounded(d, (int(w * 0.06), top, int(w * 0.94), top + block_h + 2 * pad),
             radius=int(f.size * 0.35), fill=bar)
    y = top + pad
    for ln in lines[0]:
        _text_with_shadow(d, (w / 2, y), ln, f, fg, offset=3, anchor="ma")
        y += lh
    return img


def _t_lower_third(p: dict, w: int, h: int) -> Image.Image:
    """Riga in basso: titolo + sottotitolo, pill scura."""
    img, d = _canvas(w, h)
    title = p.get("title", "Nome Cognome")
    sub = p.get("subtitle", "sottotitolo")
    bg = _hex(p.get("bg", "#000000"), int(255 * float(p.get("bg_opacity", 0.55))))
    fg = _hex(p.get("color", "#ffffff"))
    accent = _hex(p.get("accent", "#ff0050"))
    ft = _fit_font(_TEXT, title, int(w * 0.7), int(h * 0.05), weight=800)
    fs = text_font(int(h * 0.030), weight=500)
    bx0, by0, bx1 = int(w * 0.06), int(h * 0.80), int(w * 0.94)
    by1 = by0 + int(ft.size * 1.5) + int(fs.size * 1.6) + int(h * 0.03)
    _rounded(d, (bx0, by0, bx1, by1), radius=int(h * 0.02), fill=bg)
    d.rectangle((bx0, by0, bx0 + int(w * 0.012), by1), fill=accent)
    tx = bx0 + int(w * 0.05)
    d.text((tx, by0 + int(h * 0.015)), title, font=ft, fill=fg)
    d.text((tx, by0 + int(h * 0.015) + int(ft.size * 1.4)), sub.upper(), font=fs, fill=accent)
    return img


def _t_handle(p: dict, w: int, h: int) -> Image.Image:
    """@username in una pill, angolo o centro."""
    img, d = _canvas(w, h)
    handle = p.get("text", "@username")
    if not handle.startswith("@"):
        handle = "@" + handle
    fg = _hex(p.get("color", "#ffffff"))
    bg = _hex(p.get("bg", "#000000"), int(255 * float(p.get("bg_opacity", 0.45))))
    f = text_font(int(h * 0.032), weight=700)
    tw = d.textlength(handle, font=f)
    padx, pady = int(f.size * 0.7), int(f.size * 0.42)
    bw, bh = tw + 2 * padx, f.size + 2 * pady
    pos = p.get("pos", "bottom")
    x = (w - bw) / 2
    y = {"top": h * 0.06, "bottom": h * 0.88, "center": (h - bh) / 2}.get(pos, h * 0.88)
    _rounded(d, (x, y, x + bw, y + bh), radius=int(bh / 2), fill=bg)
    d.text((x + bw / 2, y + bh / 2), handle, font=f, fill=fg, anchor="mm")
    return img


def _t_badge(p: dict, w: int, h: int) -> Image.Image:
    """Bollo circolare con testo breve: NEW, -50%, LIVE..."""
    img, d = _canvas(w, h)
    text = (p.get("text") or "NEW").upper()
    fill = _hex(p.get("bg", "#ff0050"))
    fg = _hex(p.get("color", "#ffffff"))
    r = int(min(w, h) * 0.42)
    cx, cy = w // 2, h // 2
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
    f = _fit_font(_DISPLAY, text, int(r * 1.5), int(r * 0.9))
    _text_with_shadow(d, (cx, cy), text, f, fg, offset=3, anchor="mm")
    return img


def _t_price_tag(p: dict, w: int, h: int) -> Image.Image:
    """Cartellino prezzo: valore grosso + eventuale prezzo barrato sopra."""
    img, d = _canvas(w, h)
    price = str(p.get("price", "19,99€"))
    old = p.get("old_price")
    bg = _hex(p.get("bg", "#ffe600"))
    fg = _hex(p.get("color", "#111111"))
    f = _fit_font(_DISPLAY, price, int(w * 0.78), int(h * 0.34))
    bw = d.textlength(price, font=f) + int(w * 0.12)
    bh = f.size * 1.5
    fo = text_font(max(12, int(f.size * 0.4)), weight=700) if old else None
    old_h = int(fo.size * 1.4) if old else 0
    block_h = old_h + bh
    top = (h - block_h) / 2
    if old:
        ow = d.textlength(str(old), font=fo)
        oy = top + fo.size * 0.2
        d.text((w / 2, oy), str(old), font=fo, fill=_hex("#ffffff"), anchor="ma")
        ly = oy + fo.size * 0.6
        d.line((w / 2 - ow / 2 - 6, ly, w / 2 + ow / 2 + 6, ly),
               fill=_hex("#ff0050"), width=max(3, int(fo.size * 0.14)))
    y = top + old_h
    x = (w - bw) / 2
    _rounded(d, (x, y, x + bw, y + bh), radius=int(bh * 0.18), fill=bg)
    _text_with_shadow(d, (w / 2, y + bh / 2), price, f, fg, offset=2, anchor="mm")
    return img


def _t_cta(p: dict, w: int, h: int) -> Image.Image:
    """Call-to-action: pill piena con freccia."""
    img, d = _canvas(w, h)
    text = (p.get("text") or "SEGUIMI").upper()
    bg = _hex(p.get("bg", "#ff0050"))
    fg = _hex(p.get("color", "#ffffff"))
    f = _fit_font(_DISPLAY, text, int(w * 0.62), int(h * 0.3))
    arrow = int(f.size * 0.7)
    padx = int(f.size * 0.7)
    gap = int(f.size * 0.45)
    tw_text = d.textlength(text, font=f)
    bw = padx + tw_text + gap + arrow + padx
    bh = f.size * 1.7
    x, y = (w - bw) / 2, (h - bh) / 2
    _rounded(d, (x, y, x + bw, y + bh), radius=int(bh / 2), fill=bg)
    _text_with_shadow(d, (x + padx, y + bh / 2), text, f, fg, offset=3, anchor="lm")
    ax = x + padx + tw_text + gap
    ay = y + bh / 2
    d.polygon([(ax, ay - arrow / 2), (ax, ay + arrow / 2), (ax + arrow, ay)], fill=fg)
    return img


_BUILDERS = {
    "wordmark": _t_wordmark, "title_card": _t_title_card, "lower_third": _t_lower_third,
    "handle": _t_handle, "badge": _t_badge, "price_tag": _t_price_tag, "cta": _t_cta,
}


def _wrap_display(text: str, max_w: int, start: int, max_lines: int = 3):
    """Manda a capo `text` per il font display, restituendo (righe, font)."""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    size = start
    while size > 16:
        f = display_font(size)
        words, lines, cur = text.split(), [], ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if not cur or probe.textlength(trial, font=f) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        if len(lines) <= max_lines and all(probe.textlength(ln, font=f) <= max_w for ln in lines):
            return lines, f
        size -= 3
    return [text], display_font(16)


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------
def render_template(name: str, params: dict, width: int, height: int, out_png: Path) -> Path:
    if name not in _BUILDERS:
        raise ValueError(f"Template sconosciuto: {name!r}. Disponibili: {', '.join(TEMPLATES)}")
    img = _BUILDERS[name](params or {}, width, height)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def text_mask_png(text: str, width: int, height: int, out_png: Path,
                   pad: float = 0.08, fill: str = "#ffffff") -> Path:
    """Testo pieno bianco su nero trasparente: da usare come maschera ffmpeg
    (alphamerge / maskedmerge) per far scorrere il video dentro le lettere."""
    img, d = _canvas(width, height)
    lines, f = _wrap_display(text.upper().strip() or "TESTO", int(width * (1 - 2 * pad)),
                             int(height * 0.5))
    lh = int(f.size * 1.1)
    y = (height - lh * len(lines)) // 2
    for ln in lines:
        d.text((width / 2, y), ln, font=f, fill=_hex(fill), anchor="ma")
        y += lh
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


# --- SVG grezzo ------------------------------------------------------------
_SVG_BANNED_TAGS = {"script", "foreignobject", "iframe", "audio", "video", "animate",
                    "animatetransform", "animatemotion", "set", "handler"}
_URL_ATTRS = {"href", "{http://www.w3.org/1999/xlink}href", "xlink:href"}


def sanitize_svg(svg: str) -> str:
    """Toglie script, handler `on*`, riferimenti esterni (http/file), elementi
    di animazione. L'SVG puo' arrivare da un LLM: non ci fidiamo."""
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.DOTALL)
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        raise ValueError(f"SVG non valido: {e}") from e

    def clean(el: ET.Element) -> None:
        for child in list(el):
            tag = child.tag.split("}")[-1].lower()
            if tag in _SVG_BANNED_TAGS:
                el.remove(child)
                continue
            for k in list(child.attrib):
                kl = k.split("}")[-1].lower()
                v = child.attrib[k]
                if kl.startswith("on"):
                    del child.attrib[k]
                elif kl in {a.split("}")[-1] for a in _URL_ATTRS} and re.match(r"\s*(https?:|file:|//)", v, re.I):
                    del child.attrib[k]
                elif "url(" in v and re.search(r"url\(\s*['\"]?\s*(https?:|file:|//)", v, re.I):
                    del child.attrib[k]
            clean(child)

    clean(root)
    tag = root.tag.split("}")[-1].lower()
    if tag != "svg":
        raise ValueError("La radice non e' <svg>.")
    return ET.tostring(root, encoding="unicode")


def available_renderers() -> list[str]:
    out = []
    if shutil.which("resvg"):
        out.append("resvg")
    try:
        import cairosvg  # noqa: F401
        out.append("cairosvg")
    except Exception:  # noqa: BLE001
        pass
    return out


def render_svg(svg: str, width: int, height: int, out_png: Path) -> Path:
    """Sanifica e rasterizza SVG -> PNG RGBA. Usa resvg se c'e', poi cairosvg."""
    clean = sanitize_svg(svg)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    engines = available_renderers()
    if not engines:
        raise RuntimeError(
            "Nessun rasterizzatore SVG. Installa uno di questi:\n"
            "  - resvg  (binario, migliore):  https://github.com/linebender/resvg/releases\n"
            "  - cairosvg:  pip install cairosvg"
        )
    if "resvg" in engines:
        tmp = out_png.with_suffix(".svg")
        tmp.write_text(clean, encoding="utf-8")
        try:
            subprocess.run(
                ["resvg", "--use-fonts-dir", str(_FONT_DIR), "--skip-system-fonts",
                 "-w", str(width), "-h", str(height), str(tmp), str(out_png)],
                check=True, capture_output=True,
            )
        finally:
            tmp.unlink(missing_ok=True)
        return out_png
    import cairosvg
    cairosvg.svg2png(bytestring=clean.encode("utf-8"), write_to=str(out_png),
                     output_width=width, output_height=height)
    return out_png


def render_asset(spec: AssetSpec, out_dir: Path) -> Path:
    """Dispatch dichiarativo: AssetSpec -> PNG su disco."""
    out_dir = Path(out_dir)
    out_png = out_dir / f"{spec.asset_id}.png"
    if spec.kind == "template":
        return render_template(spec.template, spec.params, spec.width, spec.height, out_png)
    if spec.kind == "svg":
        return render_svg(spec.svg, spec.width, spec.height, out_png)
    if spec.kind == "text_mask":
        return text_mask_png(spec.text, spec.width, spec.height, out_png)
    raise ValueError(f"AssetSpec.kind sconosciuto: {spec.kind!r}")


def capabilities() -> dict:
    return {
        "templates": list(TEMPLATES),
        "svg_renderers": available_renderers(),
        "fonts": {"display": _DISPLAY.name, "text": _TEXT.name},
        "kinds": ["template", "svg", "text_mask"],
    }


# ---------------------------------------------------------------------------
# Demo:  python -m autoedit.assets --out ./_assets_demo
# ---------------------------------------------------------------------------
def _demo(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    demos = [
        AssetSpec(asset_id="wordmark", template="wordmark",
                  params={"text": "moto life", "accent": "#ff0050"}, width=1080, height=420),
        AssetSpec(asset_id="title_card", template="title_card",
                  params={"text": "come mi hanno rubato la moto", "bg": "#ff0050"},
                  width=1080, height=1920),
        AssetSpec(asset_id="lower_third", template="lower_third",
                  params={"title": "Milano, ore 3:00", "subtitle": "zona Navigli", "accent": "#ff0050"},
                  width=1080, height=1920),
        AssetSpec(asset_id="handle", template="handle",
                  params={"text": "@opodark", "pos": "bottom"}, width=1080, height=1920),
        AssetSpec(asset_id="badge", template="badge", params={"text": "-40%"}, width=600, height=600),
        AssetSpec(asset_id="price_tag", template="price_tag",
                  params={"price": "1.900€", "old_price": "2.500€"}, width=900, height=600),
        AssetSpec(asset_id="cta", template="cta", params={"text": "seguimi"}, width=1000, height=400),
        AssetSpec(asset_id="text_mask", kind="text_mask", text="MOTO", width=1080, height=1920),
    ]
    for s in demos:
        p = render_asset(s, out_dir)
        print(f"  {p.name:16s} {s.width}x{s.height}")
    print(f"\nfatto -> {out_dir}")
    print("capabilities:", capabilities())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Genera i PNG di demo dei template asset.")
    ap.add_argument("--out", type=Path, default=Path("_assets_demo"))
    _demo(ap.parse_args().out)
