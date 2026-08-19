"""paper/figures/exposure_criterion.pdf from exposure_criterion_overview.svg.

The SVG at the repository root is the author-drawn source of truth. PyMuPDF
converts it to the vector PDF the paper includes, but its SVG parser drops
<marker> elements, so every arrowhead would vanish in the conversion. All of
this figure's arrows point straight down and are attached via marker-end, so
the fix is mechanical: draw each arrowhead as an explicit chevron path at the
element's endpoint before converting. Run from the repository root.
"""

import re
from pathlib import Path

import fitz

SRC = Path("exposure_criterion_overview.svg")
OUT = Path("paper/figures/exposure_criterion.pdf")

# Stroke colours by marker id, matching the <marker> definitions in the SVG.
COLOURS = {"arrow": "#73726c", "arrow-coral": "#D85A30"}

# The markers are 6px wide chevrons (viewBox 10 scaled by 0.6): tip at the
# line end, wings 3.6px back and 2.4px out, stroked at 0.9px.
def chevron(x, y, colour):
    return (f'<path d="M{x - 2.4:.1f} {y - 3.6:.1f} L{x:.1f} {y:.1f} '
            f'L{x + 2.4:.1f} {y - 3.6:.1f}" fill="none" stroke="{colour}" '
            f'stroke-width="0.9" stroke-linecap="round" '
            f'stroke-linejoin="round"/>')


def endpoint(element):
    """The (x, y) a marker-end sits at: x2/y2 of a line, or the last
    coordinate pair of a path's d attribute."""
    if element.startswith("<line"):
        return (float(re.search(r'x2="([\d.]+)"', element).group(1)),
                float(re.search(r'y2="([\d.]+)"', element).group(1)))
    d = re.search(r'd="([^"]+)"', element).group(1)
    x, y = re.findall(r"([\d.]+)[ ,]+([\d.]+)", d)[-1]
    return float(x), float(y)


def main():
    svg = SRC.read_text()
    heads = []
    for m in re.finditer(r'<(?:line|path)\b[^>]*marker-end="url\(#([\w-]+)\)"[^>]*/>',
                         svg):
        x, y = endpoint(m.group(0))
        heads.append(chevron(x, y, COLOURS[m.group(1)]))
    if not heads:
        raise SystemExit(f"no marker-end elements found in {SRC}")
    svg = svg.replace("</svg>", "\n".join(heads) + "\n</svg>")
    OUT.write_bytes(fitz.open(stream=svg.encode(), filetype="svg").convert_to_pdf())
    print(f"wrote {OUT} with {len(heads)} arrowheads drawn explicitly")


if __name__ == "__main__":
    main()
