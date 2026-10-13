"""draw.io version of the overview figure (layout of the TikZ sketch) with the assets/overview images embedded."""
import base64
import io
import os
from xml.sax.saxutils import escape, quoteattr

from PIL import Image

SRC = "assets/overview"
OUT = os.path.join(SRC, "overview.drawio")
cells, n = [], [1]


def nid():
    n[0] += 1
    return f"c{n[0]}"


def data_uri(name, w):
    im = Image.open(os.path.join(SRC, name))
    s = 2 * w / im.width
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    if im.mode == "RGBA":
        im.save(buf, "PNG", optimize=True)
        kind = "png"
    else:
        im.convert("RGB").save(buf, "JPEG", quality=85)
        kind = "jpeg"
    return f"data:image/{kind},{base64.b64encode(buf.getvalue()).decode()}", im.height / im.width


def vertex(style, x, y, w, h, value=""):
    i = nid()
    cells.append(f'<mxCell id="{i}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def img(name, x, y, w):
    uri, r = data_uri(name, w)
    return vertex(f"shape=image;aspect=fixed;imageAspect=0;image={uri};", x, y, w, round(w * r)), round(w * r)


BOX = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=14;strokeColor=#333333;fillColor={};"


def box(text, sub, x, y, w, h=50, fill="#FFFFFF"):
    v = f"{escape(text)}<br><font style='font-size:11px' color='#555555'>{escape(sub)}</font>" if sub else escape(text)
    return vertex(BOX.format(fill), x, y, w, h, v)


def stage(title, x, y, w, h):
    vertex("rounded=1;arcSize=2;dashed=1;html=1;fillColor=none;strokeColor=#999999;strokeWidth=1.5;", x, y, w, h)
    vertex("text;html=1;align=left;verticalAlign=bottom;fontStyle=1;fontSize=15;fontFamily=Helvetica;", x, y - 30, w, 28, title)


def edge(s, t, label="", dashed=False, color="#222222", exit=None, entry=None):
    st = f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=blockThin;endFill=1;strokeWidth=2;strokeColor={color};fontSize=12;"
    st += "dashed=1;" if dashed else ""
    for k, p in (("exit", exit), ("entry", entry)):
        if p:
            st += f"{k}X={p[0]};{k}Y={p[1]};{k}Dx=0;{k}Dy=0;"
    i = nid()
    cells.append(f'<mxCell id="{i}" value={quoteattr(label)} style={quoteattr(st)} edge="1" parent="1" source="{s}" target="{t}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')


GREY, BLUE, PURPLE, ORANGE = "#EEEEEE", "#DCE8F7", "#EFE3F5", "#FCE7D6"

stage("Ingest (once per scan, Sec. III)", 20, 50, 1170, 900)
scan = box("3D scan", "XYZ", 40, 80, 220, fill=GREY)
img("01_scan_xyz.png", 40, 140, 220)
frames = box("posed frames", "optional", 40, 470, 220, fill=GREY)
for k in range(4):
    img(f"03_frame_{k + 1}.png", 40 + (k % 2) * 112, 530 + (k // 2) * 86, 108)
img("04_frames_poses.png", 40, 710, 220)

enc = box("encode", "segmenter + dense 3D", 320, 80, 360)
img("06_encode_prompts.png", 320, 140, 175)
img("05_encode_dense.png", 505, 140, 175)
prop = box("propose objects", "prompts + lifted 2D", 320, 330, 360)
img("07_proposals.png", 400, 390, 200)
imgv = box("image evidence", "multi-view crops", 320, 640, 360)
x = 320
for k in range(3):
    im = Image.open(os.path.join(SRC, f"08_image_crop_{k + 1}.png"))
    w = round(150 * im.width / im.height)
    img(f"08_image_crop_{k + 1}.png", x, 700, w)
    x += w + 10

mat = box("materialize", "one tuple / object", 760, 330, 400)
img("09_materialized.png", 780, 390, 360)

edge(scan, enc, exit=(1, 0.5), entry=(0, 0.5))
edge(scan, prop, exit=(1, 0.5), entry=(0, 0.3))
edge(frames, imgv, color="#888888", exit=(1, 0.5), entry=(0, 0.5))
edge(frames, prop, color="#888888", exit=(1, 0.5), entry=(0, 0.75))
edge(enc, mat, exit=(1, 0.5), entry=(0.3, 0))
edge(prop, mat, exit=(1, 0.5), entry=(0, 0.5))
edge(imgv, mat, color="#888888", exit=(1, 0.5), entry=(0.3, 1))

rows = [("oid", "geometry", "token", "dense", "image", "prov."), ("1", "mask, box, level", "t<sub>1</sub>", "f<sub>1</sub>", "v<sub>1</sub>", "auto"),
        ("2", "mask, box, level", "t<sub>2</sub>", "f<sub>2</sub>", "&empty;", "auto"), ("&vellip;", "", "", "", "", ""),
        ("M+1", "mask, box, level", "t", "f", "v", "query")]
tbl = "<table style='border-collapse:collapse;font-size:13px;font-family:Helvetica'>" + "".join(
    "<tr>" + "".join(f"<td style='padding:3px 7px;{'font-weight:bold;border-bottom:1px solid #333' if r == 0 else ''}'>{c}</td>" for c in row) + "</tr>"
    for r, row in enumerate(rows)) + "</table>"
vertex("text;html=1;align=left;verticalAlign=bottom;fontStyle=1;fontSize=15;fontFamily=Helvetica;", 1230, 420, 400, 28, "Object relation R")
rel = vertex(f"rounded=1;arcSize=4;html=1;whiteSpace=wrap;fillColor={BLUE};strokeColor=#333333;align=center;", 1230, 450, 440, 170, tbl)
edge(mat, rel, exit=(1, 0.5), entry=(0, 0.5))

stage("OVQL engine (Secs. IV-V)", 1730, 50, 700, 900)
qry = vertex(BOX.format(PURPLE) + "fontFamily=Courier New;fontSize=13;", 1760, 80, 640, 60,
             "SELECT WHERE IS('chair') AND<br>NEAR(IS('table'), 0.5)")
p1 = box("phase 1", "refinement planning", 1760, 200, 290)
p2 = box("phase 2", "scoring + joins", 2110, 200, 290)
img("10_query_heat_is.png", 2130, 260, 250)
res = box("ranked (oid, P)", "", 2110, 520, 290, fill=ORANGE)
img("11_query_join_ranked.png", 2130, 580, 250)
ref = box("refine → verify → admit", "", 1760, 520, 290, fill=GREY)
b, _ = img("12_refine_before.png", 1760, 600, 135)
a, _ = img("13_refine_after.png", 1915, 600, 135)
edge(b, a, exit=(1, 0.5), entry=(0, 0.5))

edge(qry, p1, exit=(0.25, 1), entry=(0.5, 0))
edge(p1, p2, exit=(1, 0.5), entry=(0, 0.5))
edge(p2, res, exit=(1, 0.5), entry=(1, 0.5))
edge(p1, ref, "no object explains c", exit=(0.5, 1), entry=(0.5, 0))
edge(rel, p1, exit=(1, 0.3), entry=(0, 0.5))
edge(ref, rel, "semantic cracking", dashed=True, color="#3A5E9C", exit=(0, 0.5), entry=(0.7, 1))

xml = ('<mxfile host="drawio"><diagram name="Overview" id="overview"><mxGraphModel dx="2400" dy="1000" grid="1" gridSize="10" '
       'page="1" pageWidth="2460" pageHeight="980" background="#FFFFFF"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
open(OUT, "w").write(xml)
print(OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB", len(cells), "cells")
