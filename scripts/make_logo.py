"""HgRT logo: a drop of liquid mercury (Hg, the element the runtime is named after) pinching off into two, and a silver
wordmark with a sheen. The drop is two circles joined by a metaball neck (cubic Beziers tangent to both circles), drawn
as one path so the union has no seams, with a silver gradient, a soft shade at the rim and a highlight per lobe.

Writes docs/assets/hgrt-{light,dark}.svg / .png and hgrt-mark-{light,dark}.svg / .png. Text is converted to outline
paths (IBM Plex Sans SemiBold through fontTools), so the SVGs need no installed font. PNGs are rendered with a
headless Chromium (HGRT_CHROME, default: Playwright's chrome-headless-shell, else google-chrome).

    python scripts/make_logo.py --font /path/to/IBMPlexSans[wdth,wght].ttf
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
BLOB = {"light": ["#c9d0d7", "#8e98a3", "#5d6772"], "dark": ["#dfe4e9", "#a3acb6", "#6b747e"]}
# the two lobes (centre, radius) and the neck spread v: lower v = thinner neck, i.e. closer to separating
C1, R1, C2, R2, SPREAD = (68, 118), 50, (152, 46), 27, 0.36
SHEEN = {"light": ["#56606a", "#8e98a3", "#e3e8ed", "#8e98a3", "#56606a"],
         "dark": ["#8f99a3", "#bcc5ce", "#f7f9fb", "#bcc5ce", "#8f99a3"]}
MARK_W, H, WORD_X = 190, 190, 214


def _vec(c, a, r):
    return c[0] + r * math.cos(a), c[1] + r * math.sin(a)


def neck(c1, r1, c2, r2, v, handle=2.4):
    """the metaball bridge between two circles: tangent points spread by v, joined by cubic Beziers."""
    d = math.dist(c1, c2)
    u1 = u2 = 0.0
    if d < r1 + r2:
        u1 = math.acos((r1 * r1 + d * d - r2 * r2) / (2 * r1 * d))
        u2 = math.acos((r2 * r2 + d * d - r1 * r1) / (2 * r2 * d))
    ab = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
    spread = math.acos((r1 - r2) / d)
    a1, a2 = ab + u1 + (spread - u1) * v, ab - u1 - (spread - u1) * v
    a3 = ab + math.pi - u2 - (math.pi - u2 - spread) * v
    a4 = ab - math.pi + u2 + (math.pi - u2 - spread) * v
    p1, p2, p3, p4 = _vec(c1, a1, r1), _vec(c1, a2, r1), _vec(c2, a3, r2), _vec(c2, a4, r2)
    k = min(v * handle, math.dist(p1, p3) / (r1 + r2)) * min(1, d * 2 / (r1 + r2))
    h1, h2 = _vec(p1, a1 - math.pi / 2, r1 * k), _vec(p2, a2 + math.pi / 2, r1 * k)
    h3, h4 = _vec(p3, a3 + math.pi / 2, r2 * k), _vec(p4, a4 - math.pi / 2, r2 * k)
    f = lambda q: f"{q[0]:.2f} {q[1]:.2f}"
    return f"M {f(p1)} C {f(h1)} {f(h3)} {f(p3)} A {r2} {r2} 0 0 0 {f(p4)} C {f(h4)} {f(h2)} {f(p2)} Z"


def blob(mode):
    """the drop: one path (both circles + the neck), gradient fill, rim shade and a highlight on each lobe."""
    circle = lambda c, r: f"M {c[0] - r} {c[1]} a {r} {r} 0 1 0 {2 * r} 0 a {r} {r} 0 1 0 {-2 * r} 0 Z "
    d = circle(C1, R1) + circle(C2, R2) + neck(C1, R1, C2, R2, SPREAD)
    g = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in zip((0, .5, 1), BLOB[mode]))

    def highlight(c, r):
        x, y = c[0] - r * .32, c[1] - r * .38
        return (f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{r * .42:.1f}" ry="{r * .26:.1f}" '
                f'transform="rotate(-35 {x:.1f} {y:.1f})" fill="url(#hl)"/>')
    defs = (f'<linearGradient id="hg" x1="0.15" y1="0" x2="0.85" y2="1">{g}</linearGradient>'
            '<radialGradient id="hl"><stop offset="0" stop-color="#ffffff" stop-opacity=".95"/>'
            '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></radialGradient>'
            '<radialGradient id="rim" cx="0.5" cy="0.15" r="0.9"><stop offset="0.55" stop-color="#000" stop-opacity="0"/>'
            '<stop offset="1" stop-color="#000" stop-opacity=".22"/></radialGradient>'
            f'<clipPath id="drop"><path d="{d}"/></clipPath>')
    body = (f'<path d="{d}" fill="url(#hg)"/><g clip-path="url(#drop)">'
            f'<rect width="{MARK_W}" height="{H}" fill="url(#rim)"/>{highlight(C1, R1)}{highlight(C2, R2)}</g>')
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


def svg(mode, full, font, scale=1):
    bdefs, bbody = blob(mode)
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
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        for full, stem in ((True, "hgrt"), (False, "hgrt-mark")):
            text, W = svg(mode, full, a.font)
            (OUT / f"{stem}-{mode}.svg").write_text(text)
            render(svg(mode, full, a.font, scale=3)[0], str((OUT / f"{stem}-{mode}.png").resolve()), W * 3, H * 3)
            print("wrote", OUT / f"{stem}-{mode}.png")
