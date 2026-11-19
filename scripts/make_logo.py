"""HgRT logo: a drop of liquid mercury (Hg, the element the runtime is named after) pinching off into two, drawn in
ASCII like the Minerva site's mercury background (assets/js/mercury-bg.js), and a silver wordmark with a sheen.

The drop is two metaballs with a sharp (r/d)^4 falloff, so both stay round and only a thin neck (~10 px) joins them,
shaded as a chrome height field (environment bands, Fresnel, one specular light) and quantized to the glyph ramp
" .,:-=+*oO%#@". Glyphs are DejaVu Sans Mono Bold outlines (one <path> per glyph, placed with <use>), so the SVGs need no
installed font either.

Writes docs/assets/hgrt-{light,dark}.svg / .png and hgrt-mark-{light,dark}.svg / .png. Text is converted to outline
paths (IBM Plex Sans SemiBold through fontTools), so the SVGs need no installed font. PNGs are rendered with a
headless Chromium (HGRT_CHROME, default: Playwright's chrome-headless-shell, else google-chrome).

    python scripts/make_logo.py --font /path/to/IBMPlexSans[wdth,wght].ttf --mono /path/to/DejaVuSansMono-Bold.ttf
"""
import argparse
import glob
import math
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
# silver: base, and the sheen (dark at the ends, a bright band in the middle), per background
SILVER = {"light": "#8e98a3", "dark": "#b9c2cb"}
# the drop: two metaballs (x, y, r) in the 190 x 190 mark; with FALLOFF 4 their 12.75 px gap leaves a ~10 px neck
BALLS = [(66.7, 105.4, 47), (143.3, 64.6, 27)]
FALLOFF = 4
RAMP = " .,:-=+*oO%#@"
CELL = 5.0                                   # glyph size in px (the site background uses 9)
INK = {"light": ("#6b7580", "#2b3138"), "dark": ("#9aa4ae", "#f4f7fa")}   # body ink, brightest-level ink
SHEEN = {"light": ["#56606a", "#8e98a3", "#e3e8ed", "#8e98a3", "#56606a"],
         "dark": ["#8f99a3", "#bcc5ce", "#f7f9fb", "#bcc5ce", "#8f99a3"]}
MARK_W, H, WORD_X = 190, 190, 214


def _field(x, y):
    import numpy as np
    f = np.zeros_like(x)
    for bx, by, br in BALLS:
        f += (br * br / ((x - bx) ** 2 + (y - by) ** 2 + 1)) ** (FALLOFF / 2)
    return f


def ascii_drop(t=7.3):
    """per cell: (x, y, v) with v the shaded brightness in [0, 1] (0 = empty). The shading is mercury-bg.js's: the field
    as a height map, normals by finite differences plus a slow ripple, environment bands, Fresnel and a specular light;
    the body is lifted (0.22 + 0.95 v) so the drop reads solid at logo size."""
    import numpy as np
    cw, ch = CELL * 0.602, CELL * 1.08
    cols, rows = int(MARK_W / cw), int(H / ch)
    x = (np.arange(cols) + .5)[None, :] * cw + (MARK_W - cols * cw) / 2
    y = (np.arange(rows) + .5)[:, None] * ch + (H - rows * ch) / 2
    x, y = np.broadcast_arrays(x, y)
    e, R = cw, min(MARK_W, H) * 0.11
    z = lambda ff: np.sqrt(np.clip(1 - 1 / ff, 0, None))
    f = _field(x, y)
    z0, zx, zy = z(f), z(_field(x + e, y)), z(_field(x, y + e))
    nx = -(zx - z0) / e * R + 0.18 * np.sin(x * 0.021 + y * 0.007 + t * 1.7)
    ny = -(zy - z0) / e * R + 0.18 * np.cos(y * 0.026 - x * 0.006 - t * 1.3)
    nl = np.sqrt(nx ** 2 + ny ** 2 + 1)
    nx, ny, nz = nx / nl, ny / nl, 1 / nl
    lx, ly, lz = 0.55 * math.cos(t * .35), -0.6, 0.6
    ll = math.sqrt(lx * lx + ly * ly + lz * lz)
    rx, ry = 2 * nz * nx, 2 * nz * ny
    env = 0.5 + 0.5 * np.sin(ry * 7 + rx * 2.5 + t * .6)
    env = np.where(ry < 0, .35 + .55 * env, .12 + .25 * env)
    spec = np.maximum(0, (nx * lx + ny * ly + nz * lz) / ll) ** 28
    v = np.minimum(1, .1 + env * .7 + (1 - nz) ** 2 * .35 + spec * .8)
    v = np.where(f < 1, v * np.clip((f - .92) / .08, 0, 1) * .6, np.minimum(1, 0.22 + 0.95 * v))
    v[f < .92] = 0
    return x - cw / 2, y - ch / 2, v


def glyphs(mono):
    """the ramp's glyphs as outline paths at CELL px, origin at the cell's top-left."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont
    font = TTFont(mono)
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    sc = CELL / font["head"].unitsPerEm
    asc = font["hhea"].ascent * sc
    out = {}
    for i, c in enumerate(RAMP[1:], 1):
        pen = SVGPathPen(gs)
        gs[cmap[ord(c)]].draw(TransformPen(pen, (sc, 0, 0, -sc, 0, asc)))
        out[i] = pen.getCommands()
    return out


def blob(mode, mono):
    """the ASCII drop: six brightness levels (the brightest in the highlight ink), glyph by brightness."""
    X, Y, V = ascii_drop()
    levels, ink, hi = 6, *INK[mode]
    defs = "".join(f'<path id="a{i}" d="{d}"/>' for i, d in glyphs(mono).items())
    by_level = {}
    for x, y, v in zip(X.ravel(), Y.ravel(), V.ravel()):
        k, gi = round(v * levels), min(len(RAMP) - 1, round(v * (len(RAMP) - 1)))
        if k > 0 and gi > 0:
            by_level.setdefault(k, []).append(f'<use href="#a{gi}" x="{x:.1f}" y="{y:.1f}"/>')
    body = "".join(f'<g fill="{hi if k >= levels else ink}" fill-opacity="{1.0 if k >= levels else 0.55 + 0.45 * k / levels:.2f}">'
                   f'{"".join(u)}</g>' for k, u in sorted(by_level.items()))
    return defs, body


def text_path(font_path, text, size, x, y, tracking=-2.0, weight=600):
    """text -> one SVG path (outlines), baseline at y; returns (d, advance width in px)."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont
    font = TTFont(font_path)
    if "fvar" in font:
        from fontTools.varLib.instancer import instantiateVariableFont
        axes = {a.axisTag: (weight if a.axisTag == "wght" else a.defaultValue) for a in font["fvar"].axes}
        font = instantiateVariableFont(font, axes)
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    pen = SVGPathPen(gs)
    cx = x
    for ch in text:
        g = cmap[ord(ch)]
        gs[g].draw(TransformPen(pen, (scale, 0, 0, -scale, cx, y)))
        cx += gs[g].width * scale + tracking
    return pen.getCommands(), cx - tracking - x


def svg(mode, full, font, mono, scale=1):
    bdefs, bbody = blob(mode, mono)
    sh = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in zip((0, .38, .5, .62, 1), SHEEN[mode]))
    parts = [f'<defs>{bdefs}<linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">{sh}</linearGradient></defs>',
             bbody]
    W = MARK_W
    if full:
        d, adv = text_path(font, "HgRT", 96, WORD_X, 128)
        parts.append(f'<path d="{d}" fill="url(#sheen)"/>')
        W = int(WORD_X + adv + 18)
    body = "\n  ".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * scale}" '
            f'height="{H * scale}">\n  {body}\n</svg>\n'), W


def chrome():
    env = os.environ.get("HGRT_CHROME")
    if env:
        return env
    found = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell"))
                   + glob.glob("/mnt/ext/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell"))
    return found[-1] if found else "google-chrome"


def render(svg_text, png_path, w, h):
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(svg_text)
        src = f.name
    subprocess.run([chrome(), "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    "--default-background-color=00000000", f"--window-size={w},{h}",
                    f"--screenshot={png_path}", f"file://{src}"], check=True, capture_output=True, timeout=120)
    os.unlink(src)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default=os.path.expanduser("~/.local/share/fonts/IBMPlexSans-Variable.ttf"))
    ap.add_argument("--mono", default="/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        for full, stem in ((True, "hgrt"), (False, "hgrt-mark")):
            text, W = svg(mode, full, a.font, a.mono)
            (OUT / f"{stem}-{mode}.svg").write_text(text)
            render(svg(mode, full, a.font, a.mono, scale=3)[0], str((OUT / f"{stem}-{mode}.png").resolve()), W * 3, H * 3)
            print("wrote", OUT / f"{stem}-{mode}.png")
