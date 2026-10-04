"""OpenSNAP system explainer. Render: manim -qh docs/video/opensnap_explainer.py OpenSNAPExplainer"""
import os

from manim import *

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
BG, INK, MUTED = "#0f1419", "#e6edf3", "#8b949e"
BLUE, GREEN, ORANGE, RED, PURPLE = "#58a6ff", "#3fb950", "#f0883e", "#ff7b72", "#d2a8ff"
FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"


def T(s, size=28, color=INK, font=FONT, weight=NORMAL):
    return Text(s, font=font, font_size=size, color=color, weight=weight)


def box(label, color=BLUE, w=2.6, h=0.9, size=20):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color, stroke_width=2.5,
                         fill_color=color, fill_opacity=0.12)
    return VGroup(r, T(label, size).move_to(r))


def pic(name, h=4.2):
    return ImageMobject(os.path.join(IMG, f"{name}.png")).set_height(h)


class OpenSNAPExplainer(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.title()
        self.problem()
        self.ingest()
        self.query()
        self.interact()
        self.results()
        self.outro()

    def clear_all(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def title(self):
        t = T("OpenSNAP", 72, weight=BOLD)
        s = T("Open-vocabulary querying over 3D point clouds", 32, MUTED).next_to(t, DOWN, 0.4)
        self.play(Write(t), run_time=1.2)
        self.play(FadeIn(s, shift=UP * 0.2))
        self.wait(1.2)
        self.clear_all()

    def problem(self):
        img = pic("scene", 4.6).to_edge(LEFT, buff=0.6)
        cap = T("a raw scan: millions of XYZ points", 22, MUTED).next_to(img, DOWN, 0.2)
        self.play(FadeIn(img), FadeIn(cap))
        qs = ["where is the beanbag?", "chairs next to a table?", "what is this object?", "something to sit on"]
        bubbles = VGroup(*[T(f"“{q}”", 26, ORANGE) for q in qs]).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        bubbles.next_to(img, RIGHT, 0.8)
        for b in bubbles:
            self.play(FadeIn(b, shift=LEFT * 0.3), run_time=0.45)
        note = T("No labels. Any words. Clicks or text.", 26, INK).to_edge(DOWN, buff=0.4)
        self.play(Write(note))
        self.wait(1.2)
        self.clear_all()

    def ingest(self):
        head = T("Ingest plane · once per scene", 34, BLUE, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(head))
        pc = box("point cloud\n(XYZ)", MUTED, 2.2, 1.0)
        vox = box("voxelize\n2 cm", MUTED, 1.8, 1.0)
        bb = box("promptable\nsegmenter (frozen)", BLUE, 2.8, 1.0)
        cl = box("3D CLIP\nfeatures (frozen)", PURPLE, 2.8, 1.0)
        fs = box("feature\nstore", GREEN, 1.9, 1.0)
        VGroup(pc, vox).arrange(RIGHT, buff=0.5).move_to(LEFT * 4.3 + UP * 1.4)
        VGroup(bb, cl).arrange(DOWN, buff=0.35).next_to(vox, RIGHT, 0.7)
        fs.next_to(VGroup(bb, cl), RIGHT, 0.7)
        arrows = VGroup(Arrow(pc.get_right(), vox.get_left(), buff=0.08, color=MUTED),
                        Arrow(vox.get_right(), bb.get_left(), buff=0.08, color=MUTED),
                        Arrow(vox.get_right(), cl.get_left(), buff=0.08, color=MUTED),
                        Arrow(bb.get_right(), fs.get_left(), buff=0.08, color=MUTED),
                        Arrow(cl.get_right(), fs.get_left(), buff=0.08, color=MUTED))
        self.play(LaggedStart(*[FadeIn(m) for m in (pc, vox, bb, cl, fs)], lag_ratio=0.25), Create(arrows), run_time=2.2)
        dec = box("auto-prompt → mask decoder → NMS", BLUE, 5.4, 0.8).next_to(VGroup(bb, cl, fs), DOWN, 0.55)
        self.play(FadeIn(dec, shift=UP * 0.2), Create(Arrow(fs.get_bottom(), dec.get_top() + RIGHT * 2, buff=0.08, color=MUTED)))
        cols = ["oid", "voxels", "bbox", "conf", "token e", "CLIP p", "prov"]
        rows = [["0", "{…}", "▭", "c₀", "▮▮▮", "▮▮▮", "auto"], ["1", "{…}", "▭", "c₁", "▮▮▮", "▮▮▮", "auto"],
                ["2", "{…}", "▭", "c₂", "▮▮▮", "▮▮▮", "auto"]]
        table = self.make_table(cols, rows).scale(0.8).to_edge(DOWN, buff=0.35)
        lab = T("object relation", 22, GREEN).next_to(table, LEFT, 0.3)
        self.play(FadeIn(lab), FadeIn(table[0]))
        for r in table[1:]:
            self.play(FadeIn(r, shift=DOWN * 0.15), run_time=0.35)
        self.ingest_table = (table, lab)
        msg = T("semantics are not a label column: P(text | object) is computed at query time", 20, MUTED)
        msg.next_to(head, DOWN, 0.15)
        self.play(FadeIn(msg))
        self.wait(1.6)
        self.clear_all()

    def make_table(self, cols, rows, prov_color=None):
        w = [0.7, 1.7, 0.8, 0.9, 1.2, 1.2, 1.5]
        lines = []
        for j, row in enumerate([cols] + rows):
            cells = VGroup()
            x = 0
            for k, (val, ww) in enumerate(zip(row, w)):
                c = INK if j else MUTED
                if j and k == 6 and val.startswith("query"):
                    c = ORANGE
                cell = VGroup(Rectangle(width=ww, height=0.5, stroke_color="#30363d", stroke_width=1.2),
                              T(val, 18, c, MONO))
                cell[1].move_to(cell[0])
                cell.move_to(RIGHT * (x + ww / 2))
                x += ww
                cells.add(cell)
            cells.shift(DOWN * 0.5 * j)
            lines.append(cells)
        return VGroup(*lines).center()

    def query(self):
        head = T("Query plane · per query, milliseconds", 34, ORANGE, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(head))
        q = T("SELECT WHERE IS('beanbag') AND NEAR(IS('sofa'), 1.0)", 26, INK, MONO).next_to(head, DOWN, 0.45)
        self.play(AddTextLetterByLetter(q, time_per_char=0.03))
        root = box("AND", MUTED, 1.2, 0.6, 20)
        l = box("IS('beanbag')", ORANGE, 2.4, 0.6, 18)
        n = box("NEAR(·, 1.0 m)", MUTED, 2.2, 0.6, 18)
        r = box("IS('sofa')", ORANGE, 1.9, 0.6, 18)
        root.move_to(LEFT * 3.9 + UP * 0.7)
        l.next_to(root, DOWN + LEFT, 0.4)
        n.next_to(root, DOWN + RIGHT, 0.4)
        r.next_to(n, DOWN, 0.4)
        edges = VGroup(Line(root.get_bottom(), l.get_top(), color=MUTED), Line(root.get_bottom(), n.get_top(), color=MUTED),
                       Line(n.get_bottom(), r.get_top(), color=MUTED))
        plan_t = T("logical plan", 20, MUTED).next_to(root, UP, 0.2)
        self.play(FadeIn(VGroup(root, l, n, r, edges, plan_t)), run_time=1.0)
        p1 = T("① coverage: can a stored object be a 'beanbag'?", 21, INK).move_to(UP * 1.2)
        p1.shift(RIGHT * (-0.4 - p1.get_left()[0]))
        p1b = T("max P(beanbag | o) < θ  →  refine", 21, RED, MONO).next_to(p1, DOWN, 0.18).align_to(p1, LEFT)
        self.play(FadeIn(p1), FadeIn(p1b))
        crack = T("② semantic cracking: the query is the click oracle", 21, ORANGE).next_to(p1b, DOWN, 0.4).align_to(p1, LEFT)
        crack2 = T("clicks where text relevance is high, mask is missing", 17, MUTED).next_to(crack, DOWN, 0.12).align_to(p1, LEFT)
        img = pic("q_beanbag", 2.2).next_to(crack2, DOWN, 0.2).align_to(p1, LEFT)
        self.play(FadeIn(crack), FadeIn(crack2))
        self.play(FadeIn(img))
        cols = ["oid", "voxels", "bbox", "conf", "token e", "CLIP p", "prov"]
        rows = [["…", "…", "…", "…", "…", "…", "auto"], ["new", "{…}", "▭", "c", "▮▮▮", "▮▮▮", "query:beanbag"]]
        table = self.make_table(cols, rows).scale(0.62).to_corner(DL, buff=0.35)
        self.play(FadeIn(table[0]), FadeIn(table[1]))
        self.play(FadeIn(table[2], shift=DOWN * 0.2), Flash(table[2], color=ORANGE, flash_radius=1.2))
        keep = T("persisted: later queries reuse it", 17, ORANGE).next_to(table, UP, 0.12).align_to(table, LEFT)
        self.play(FadeIn(keep))
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in (p1, p1b, crack, crack2, img, keep, table)])
        p2 = T("③ probabilistic evaluation", 23, INK).move_to(UP * 1.2)
        p2.shift(RIGHT * (-0.4 - p2.get_left()[0]))
        f1 = T("IS(t)  = P(t | o) from CLIP p + token e", 19, MUTED, MONO)
        f2 = T("NEAR   = max over neighbours within 1 m", 19, MUTED, MONO)
        f3 = T("AND = ×   OR = noisy-or   NOT = 1 − p", 19, MUTED, MONO)
        VGroup(f1, f2, f3).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(p2, DOWN, 0.3).align_to(p2, LEFT)
        self.play(FadeIn(p2), LaggedStart(FadeIn(f1), FadeIn(f2), FadeIn(f3), lag_ratio=0.3))
        bars = VGroup()
        for i, (name, v) in enumerate([("new object (beanbag)", 0.85), ("other object", 0.15), ("other object", 0.08)]):
            b = Rectangle(width=4.0 * v, height=0.35, fill_color=GREEN if i == 0 else MUTED, fill_opacity=0.8, stroke_width=0)
            lbl = T(name, 18, INK).next_to(b, RIGHT, 0.15)
            bars.add(VGroup(b, lbl).arrange(RIGHT, buff=0.15, aligned_edge=ORIGIN))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(VGroup(f1, f2, f3), DOWN, 0.45).align_to(f1, LEFT)
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.25))
        self.play(FadeIn(T("(schematic)", 16, MUTED).next_to(bars, DOWN, 0.15).align_to(bars, LEFT)))
        self.wait(1.4)
        self.clear_all()

    def interact(self):
        head = T("Click an object → mask + name", 34, GREEN, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(head))
        a, b = pic("click_chair", 3.6), pic("click_table", 3.6)
        Group(a, b).arrange(RIGHT, buff=0.5).next_to(head, DOWN, 0.4)
        self.play(FadeIn(a))
        self.play(FadeIn(b))
        s1 = T("3D evidence proposes top-5 names from an open vocabulary", 22, INK)
        s2 = T("a small VLM chooses among them from renders of the object", 22, INK)
        s3 = T("the clicked mask is the segmenter's own frozen output", 22, MUTED)
        VGroup(s1, s2, s3).arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.45)
        self.play(LaggedStart(FadeIn(s1), FadeIn(s2), FadeIn(s3), lag_ratio=0.4))
        self.wait(1.6)
        self.clear_all()

    def results(self):
        head = T("Measured", 34, INK, weight=BOLD).to_edge(UP, buff=0.35)
        self.play(Write(head))
        title = T("Open-vocab 3D segmentation, ScanNet val (mIoU)", 21, MUTED).move_to(UP * 2.1).to_edge(LEFT, buff=0.6)
        data = [("segmenter text head", 41.5, MUTED), ("OpenScene", 52.0, PURPLE), ("OpenSNAP", 58.2, GREEN)]
        bars = VGroup()
        for name, v, c in data:
            r = Rectangle(width=v / 12, height=0.55, fill_color=c, fill_opacity=0.85, stroke_width=0)
            g = VGroup(T(name, 20, INK), r, T(f"{v:.1f}", 20, INK))
            g[0].scale_to_fit_width(min(g[0].width, 2.7))
            bars.add(g.arrange(RIGHT, buff=0.25))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(title, DOWN, 0.35).align_to(title, LEFT)
        self.play(FadeIn(title), LaggedStart(*[FadeIn(b, shift=RIGHT * 0.3) for b in bars], lag_ratio=0.3))
        stats = VGroup(
            T("query on stored relation", 20, MUTED), T("0.8 ms", 40, GREEN, weight=BOLD),
            T("vs re-processing the scan", 20, MUTED), T("8.8 s", 40, RED, weight=BOLD),
            T("68-room catalog vs raw XYZ", 20, MUTED), T("171 MB vs 945 MB", 30, BLUE, weight=BOLD),
            T("unseen-class naming vs segmenter", 20, MUTED), T("17.2% vs 12.2%", 30, ORANGE, weight=BOLD),
        ).arrange(DOWN, buff=0.12).move_to(RIGHT * 3.6 + DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(s) for s in stats], lag_ratio=0.12), run_time=2.2)
        self.wait(2.0)
        self.clear_all()

    def outro(self):
        t = T("OpenSNAP", 60, weight=BOLD)
        s = T("data model · OVQL · semantic cracking · click-to-name", 26, MUTED).next_to(t, DOWN, 0.35)
        u = T("github.com/suhanpahari/OpenSNAP", 26, BLUE, MONO).next_to(s, DOWN, 0.5)
        self.play(FadeIn(t), FadeIn(s))
        self.play(Write(u))
        self.wait(2.0)
