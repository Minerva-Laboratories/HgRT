"""HgRT logo: a silver hexagon with a white mercury sign (Hg, the element the runtime is named after), and a silver
wordmark with a sheen. The same construction as Minerva's Moneta logo (gold there, silver here).

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
HEX = {"light": ["#b3bcc6", "#8e98a3", "#6f7983"], "dark": ["#a3acb6", "#848e98", "#646d77"]}
SHEEN = {"light": ["#56606a", "#8e98a3", "#e3e8ed", "#8e98a3", "#56606a"],
         "dark": ["#8f99a3", "#bcc5ce", "#f7f9fb", "#bcc5ce", "#8f99a3"]}
MARK_W, H, WORD_X = 190, 190, 214


def hexagon(cx=95, cy=95, r=84):
    return " ".join(f"{cx + r * math.cos(math.pi / 6 + k * math.pi / 3):.1f},"
                    f"{cy + r * math.sin(math.pi / 6 + k * math.pi / 3):.1f}" for k in range(6))


def mercury_sign():
    """the alchemical / astronomical sign of mercury, stroked: horns (an open cup), a circle, a cross below."""
    w = 'fill="none" stroke="#ffffff" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"'
    return (f'<path d="M 72 38 A 23 23 0 0 0 118 38" {w}/>'
            f'<circle cx="95" cy="84" r="24" {w}/>'
            f'<path d="M 95 108 L 95 152 M 76 132 L 114 132" {w}/>')


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
    hx = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in zip((0, .55, 1), HEX[mode]))
    sh = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in zip((0, .38, .5, .62, 1), SHEEN[mode]))
    parts = [f'<defs><linearGradient id="hex" x1="0" y1="0" x2="1" y2="1">{hx}</linearGradient>'
             f'<linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">{sh}</linearGradient></defs>',
             f'<polygon points="{hexagon()}" fill="url(#hex)" stroke="url(#hex)" stroke-width="10" '
             f'stroke-linejoin="round"/>', mercury_sign()]
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
