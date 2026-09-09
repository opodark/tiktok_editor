"""Maschere di ritaglio e effetti *attraverso* la maschera (stile TikTok).

Una "maschera" e' un flusso in scala di grigi: bianco = dentro la forma,
nero = fuori. La si costruisce come frammento di `filter_complex` ffmpeg,
gia' pronto a essere animato (cresce / si restringe / pulsa), ruotato e
sfumato ai bordi.

Due usi:
- `masked_effect(...)`  -> un effetto (blur, desaturazione, RGB-shift...)
  applicato SOLO dentro (o solo fuori) la forma, su una clip sola.
- `shape_reveal(...)`   -> la forma che cresce fa comparire la clip B sopra
  la clip A: la transizione "a cuore / a cerchio / dentro le lettere".

Le forme geometriche sono generate al volo con `geq` (equazione per
pixel) su una maschera a bassa risoluzione, poi riscalata e sfumata:
veloce e senza file intermedi. `text` e `image` usano `movie=`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SHAPES = ("circle", "ellipse", "rect", "rrect", "diamond", "heart", "text", "image")
GROWTH = ("in", "out", "pulse", "none")

# effetti pronti da usare col parametro `effect` di masked_effect
EFFECT_CHAINS = {
    "blur": "gblur=sigma=22",
    "desat": "hue=s=0.15",
    "bw": "hue=s=0",
    "rgbshift": "rgbashift=rh=12:bh=-12:gv=7",
    "zoompunch": "scale=iw*1.18:ih*1.18,crop=iw/1.18:ih/1.18",
    "bright": "eq=brightness=0.12:saturation=1.3",
    "dark": "eq=brightness=-0.35:saturation=0.7",
    "pixelize": "pixelize=w=24:h=24",
}


def _progress_expr(growth: str, dur: float) -> str:
    d = max(0.05, float(dur))
    return {
        "in": f"clip(T/{d:.3f},0,1)",
        "out": f"clip(1-T/{d:.3f},0,1)",
        "pulse": f"abs(sin(T/{d:.3f}*PI))",
        "none": "1",
    }.get(growth, f"clip(T/{d:.3f},0,1)")


def _shape_test(shape: str) -> str:
    """Condizione booleana geq: vera DENTRO la forma. Usa uu, vv (coord
    normalizzate gia' ruotate, centro = 0, bordo = 1) e S (raggio)."""
    if shape == "circle":
        return "lte(hypot(uu,vv),S)"
    if shape == "ellipse":
        return "lte(hypot(uu/1.4,vv/0.72),S)"
    if shape == "rect":
        return "lte(max(abs(uu),abs(vv)),S)"
    if shape == "rrect":
        return "lte(hypot(max(abs(uu)-0.55*S,0),max(abs(vv)-0.55*S,0)),0.45*S)"
    if shape == "diamond":
        return "lte(abs(uu)+abs(vv),S)"
    if shape == "heart":
        # cuore implicito: (x^2+y^2-1)^3 - x^2 y^3 <= 0, con y verso l'alto
        return ("lte(pow(pow(uu/(S*1.35),2)+pow(-vv/(S*1.35)+0.32,2)-1,3)"
                "-pow(uu/(S*1.35),2)*pow(-vv/(S*1.35)+0.32,3),0)")
    return "lte(hypot(uu,vv),S)"


def _geq_mask(shape: str, w: int, h: int, fps: int, dur: float,
              growth: str, rotate_deg: float, spin: float, feather: float,
              invert: bool, size: float, out_label: str) -> str:
    mw = 360 if w >= h else max(120, int(360 * w / h))
    mh = 360 if h >= w else max(120, int(360 * h / w))
    p = _progress_expr(growth, dur)
    a0 = float(rotate_deg) * 3.14159265 / 180.0
    ang = f"({a0:.5f}+{float(spin):.3f}*{p}*6.2831853)"
    # S: statico -> `size`; animato -> cresce da ~0 a 2.1 (copre i 4 angoli su 9:16)
    s_expr = f"{max(0.05, float(size)):.3f}" if growth == "none" else f"0.06+2.10*({p})"
    test = (_shape_test(shape)
            .replace("uu", "ld(3)").replace("vv", "ld(4)").replace("S", "ld(0)"))
    lum = (
        # coord isotrope: entrambi gli assi divisi per meta' del lato CORTO
        f"st(1,(X-W/2)/(min(W,H)/2));st(2,(Y-H/2)/(min(W,H)/2));"
        f"st(7,{ang});"
        f"st(3,ld(1)*cos(ld(7))-ld(2)*sin(ld(7)));"
        f"st(4,ld(1)*sin(ld(7))+ld(2)*cos(ld(7)));"
        f"st(0,{s_expr});"                              # S (raggio) in ld(0)
        f"if({test},255,0)"
    )
    sigma = max(0.0, float(feather)) * max(w, h) * 0.03
    chain = (
        f"color=c=black:s={mw}x{mh}:r={fps}:d={dur:.3f},format=gray,"
        f"geq=lum='{lum}',scale={w}:{h}:flags=bilinear"
    )
    if sigma > 0.3:
        chain += f",gblur=sigma={sigma:.1f}"
    if invert:
        chain += ",negate"
    return f"{chain}[{out_label}]"


def _movie_mask(path: Path, w: int, h: int, fps: int, dur: float,
                use_alpha: bool, feather: float, invert: bool, out_label: str) -> str:
    # ffmpeg (Windows): barre in avanti + due punti con backslash, tra apici singoli
    src = str(Path(path).resolve()).replace("\\", "/").replace(":", "\\:")
    if use_alpha:
        head = f"movie='{src}',format=rgba,alphaextract"
    else:
        head = f"movie='{src}',format=gray"
    # `movie` da' UN fotogramma: lo si ripete all'infinito, poi si taglia a `dur`
    chain = (f"{head},loop=loop=-1:size=1,scale={w}:{h}:flags=bilinear,"
             f"fps={fps},trim=duration={dur:.3f},setpts=PTS-STARTPTS")
    sigma = max(0.0, float(feather)) * max(w, h) * 0.03
    if sigma > 0.3:
        chain += f",gblur=sigma={sigma:.1f}"
    if invert:
        chain += ",negate"
    return f"{chain}[{out_label}]"


def build_mask(shape: str, w: int, h: int, *, fps: int = 30, duration: float = 1.0,
               params: Optional[dict] = None, out_label: str = "m") -> str:
    """Frammento di filter_complex che PRODUCE la maschera come `[out_label]`.

    params:
      growth   "in" (cresce) | "out" (si chiude) | "pulse" | "none"   [in]
      rotate   gradi di rotazione iniziale                            [0]
      spin     giri completi nel corso dell'animazione                [0]
      feather  0..1 sfumatura del bordo                               [0.03]
      invert   scambia dentro/fuori                                   [False]
      text     (shape=text) testo -> maschera con assets.text_mask_png
      image    (shape=image) percorso PNG
      use_alpha (shape=image) usa il canale alpha invece della luminanza
    """
    p = params or {}
    growth = p.get("growth", "in")
    rotate = float(p.get("rotate", 0.0))
    spin = float(p.get("spin", 0.0))
    feather = float(p.get("feather", 0.03))
    invert = bool(p.get("invert", False))
    size = float(p.get("size", 0.55))          # raggio quando growth == "none"

    if shape == "text":
        from .assets import text_mask_png
        import tempfile
        png = Path(tempfile.mkdtemp(prefix="autoedit_mask_")) / "text_mask.png"
        text_mask_png(str(p.get("text", "TESTO")), w, h, png)
        return _movie_mask(png, w, h, fps, duration, False, feather, invert, out_label)
    if shape == "image":
        img = p.get("image") or p.get("path")
        if not img or not Path(img).is_file():
            raise ValueError("shape=image richiede params['image'] = percorso di un PNG.")
        return _movie_mask(Path(img), w, h, fps, duration,
                           bool(p.get("use_alpha", True)), feather, invert, out_label)
    if shape not in SHAPES:
        raise ValueError(f"Forma sconosciuta: {shape!r}. Disponibili: {', '.join(SHAPES)}")
    return _geq_mask(shape, w, h, fps, duration, growth, rotate, spin, feather,
                     invert, size, out_label)


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------
def masked_effect(in_label: str, effect: str, shape: str, w: int, h: int, *,
                   fps: int = 30, duration: float = 1.0, params: Optional[dict] = None,
                   inside: bool = True, out_label: str = "mfx") -> str:
    """Applica `effect` alla clip `[in_label]` SOLO dentro (inside=True) o
    solo fuori la forma. `effect` e' una chiave di EFFECT_CHAINS o una
    catena ffmpeg grezza. Ritorna un filter_complex che espone `[out_label]`."""
    fx = EFFECT_CHAINS.get(effect, effect)
    mp = dict(params or {})
    mp.setdefault("growth", "none")
    mask = build_mask(shape, w, h, fps=fps, duration=duration, params=mp, out_label="_mk")
    base, over = ("_a", "_b") if inside else ("_b", "_a")   # maskedmerge: 2° stream dove il mask e' bianco
    return (
        f"[{in_label}]split[_s0][_s1];"
        f"[_s0]null[_a];"
        f"[_s1]{fx}[_b];"
        f"{mask};"
        f"[{base}][{over}][_mk]maskedmerge[{out_label}]"
    )


def shape_reveal(from_label: str, to_label: str, shape: str, w: int, h: int, *,
                  fps: int = 30, duration: float = 1.0, params: Optional[dict] = None,
                  out_label: str = "rev") -> str:
    """La forma che cresce fa comparire `[to_label]` sopra `[from_label]`.
    `[from_label]` e `[to_label]` devono avere stessa dimensione/fps/durata."""
    mp = dict(params or {})
    mp.setdefault("growth", "in")
    mask = build_mask(shape, w, h, fps=fps, duration=duration, params=mp, out_label="_mk")
    return f"{mask};[{from_label}][{to_label}][_mk]maskedmerge[{out_label}]"


def capabilities() -> dict:
    return {"shapes": list(SHAPES), "growth": list(GROWTH), "effects": list(EFFECT_CHAINS)}


# ---------------------------------------------------------------------------
# Demo:  python -m autoedit.masks --out ./_masks_demo
# ---------------------------------------------------------------------------
def _run(cmd: list) -> None:
    import subprocess
    subprocess.run(cmd, check=True, capture_output=True)


def _demo(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H, FPS, DUR = 540, 960, 30, 1.6

    # maschere animate da sole (per vederle)
    for shp in ("circle", "heart", "diamond", "rrect"):
        frag = build_mask(shp, W, H, fps=FPS, duration=DUR,
                          params={"growth": "in", "feather": 0.05, "spin": 0.15},
                          out_label="m")
        _run(["ffmpeg", "-y", "-filter_complex", frag, "-map", "[m]",
              "-t", str(DUR), "-pix_fmt", "yuv420p", str(out_dir / f"mask_{shp}.mp4"),
              "-loglevel", "error"])

    # masked_effect: b/n fuori dal cuore, colore dentro
    fc = ("[0:v]" + EFFECT_CHAINS["bright"] + "[src];" +
          masked_effect("src", "bw", "heart", W, H, fps=FPS, duration=DUR,
                        params={"growth": "none", "feather": 0.04}, inside=False, out_label="o"))
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r={FPS}:d={DUR}",
          "-filter_complex", fc, "-map", "[o]", "-t", str(DUR),
          "-pix_fmt", "yuv420p", str(out_dir / "masked_effect_heart.mp4"), "-loglevel", "error"])

    # shape_reveal: da un colore pieno a un pattern, attraverso un cerchio che cresce
    fc = shape_reveal("0:v", "1:v", "circle", W, H, fps=FPS, duration=DUR,
                      params={"growth": "in", "feather": 0.06}, out_label="o")
    _run(["ffmpeg", "-y",
          "-f", "lavfi", "-i", f"color=c=0x111318:s={W}x{H}:r={FPS}:d={DUR}",
          "-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r={FPS}:d={DUR}",
          "-filter_complex", fc, "-map", "[o]", "-t", str(DUR),
          "-pix_fmt", "yuv420p", str(out_dir / "reveal_circle.mp4"), "-loglevel", "error"])

    # reveal attraverso il testo
    fc = shape_reveal("0:v", "1:v", "text", W, H, fps=FPS, duration=DUR,
                      params={"text": "MOTO", "feather": 0.02}, out_label="o")
    _run(["ffmpeg", "-y",
          "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}:d={DUR}",
          "-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r={FPS}:d={DUR}",
          "-filter_complex", fc, "-map", "[o]", "-t", str(DUR),
          "-pix_fmt", "yuv420p", str(out_dir / "reveal_text.mp4"), "-loglevel", "error"])

    print(f"fatto -> {out_dir}")
    print("capabilities:", capabilities())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Demo delle maschere e degli effetti attraverso maschera.")
    ap.add_argument("--out", type=Path, default=Path("_masks_demo"))
    _demo(ap.parse_args().out)
