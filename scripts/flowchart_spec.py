"""ASSUME pipeline flowchart as build steps, shared by the animation and the PowerPoint.

Canvas is 16 x 9 units (16:9), origin bottom-left. Each step is a list of primitives:
    ("box", x, y, w, h, title, sub, style)   style in {"agent", "model", "output"}
    ("arrow", [(x, y), ...], double)        polyline, arrowhead at the last point (and the first if double)
    ("line", [(x, y), ...])                 dashed separator
    ("text", x, y, s, kind)                 kind in {"title", "phase", "quote", "label"}; x is the centre, y the top
"""

COLORS = {
    "ink": "1F2933", "muted": "616E7C",
    "agent": ("DBEAFE", "2563EB"), "model": ("E5E7EB", "6B7280"), "output": ("DCFCE7", "16A34A"),
    "quote": "2563EB", "rubric": "16A34A", "sep": "D1D5DB",
}

BW, BH, Y = 2.6, 1.45, 5.25          # top-row boxes
XS = [0.35, 3.45, 6.55, 9.65, 12.75]
RY, RH = 0.55, 1.45                   # run-time row


def _mid_y():
    return Y + BH / 2


def steps():
    s = []
    s.append([("text", 8.0, 8.85, "ASSUME: agentic concept discovery for scientific images", "title"),
              ("text", 6.3, 7.75, "Concept discovery (agent loop, no labels)", "phase")])
    s.append([("box", 0.35, 7.05, BW, 0.85, "DINOv3 teacher", "frozen ViT features", "model"),
              ("arrow", [(1.65, 7.02), (1.65, Y + BH + 0.03)], False),
              ("box", XS[0], Y, BW, BH, "Tri-modal sampler", "random | farthest-point |\ncollision pairs", "model")])
    agents = [
        ("Proposer agent", "writes atomic concepts\nwith 0 / 1 / 3 / 5 anchors", "\"these two differ in\nparticle density\""),
        ("Merger agent", "prunes dead concepts;\nmerges duplicates", "\"is this different from\nparticle area fraction?\""),
        ("Verifier agent", "specimen morphology or\nimaging condition?", "\"is this about the specimen\nor about the image?\""),
    ]
    for k, (title, sub, quote) in enumerate(agents, start=1):
        step = [("arrow", [(XS[k - 1] + BW + 0.03, _mid_y()), (XS[k] - 0.03, _mid_y())], False),
                ("box", XS[k], Y, BW, BH, title, sub, "agent"),
                ("text", XS[k] + BW / 2, Y - 0.1, quote, "quote")]
        if k == 3:
            step += [("box", 9.65, 7.05, BW, 0.85, "Literature search", "Valency Bond MCP", "model"),
                     ("arrow", [(10.95, 7.02), (10.95, Y + BH + 0.03)], True)]
        s.append(step)
    s.append([("arrow", [(XS[3] + BW + 0.03, _mid_y()), (XS[4] - 0.03, _mid_y())], False),
              ("box", XS[4], Y, BW + 0.3, BH, "Concept dictionary", "named rubrics +\nliterature references", "output")])
    s.append([("line", [(0.35, 3.45), (15.65, 3.45)]),
              ("text", 2.1, 3.3, "Embedding (run time)", "phase"),
              ("box", 0.35, RY, BW, RH, "New micrograph", "unlabelled, any class\nor magnification", "model")])
    s.append([("arrow", [(2.98, RY + RH / 2), (3.97, RY + RH / 2)], False),
              ("box", 4.0, RY, 2.9, RH, "Embedding agent", "scores the micrograph\nagainst every rubric (0-5)", "agent"),
              ("text", 4.95, 2.6, "\"it has particles: describe it\nwith particle density\"", "quote"),
              ("arrow", [(15.65, _mid_y()), (15.85, _mid_y()), (15.85, 2.8), (6.6, 2.8), (6.6, RY + RH + 0.03)], False),
              ("text", 11.6, 3.15, "rubrics from the dictionary", "label")])
    s.append([("arrow", [(6.93, RY + RH / 2), (8.02, RY + RH / 2)], False),
              ("box", 8.05, RY, 3.0, RH, "Concept vector", "every dimension is a\nnamed concept", "output")])
    s.append([("arrow", [(11.08, RY + RH / 2), (12.17, RY + RH / 2)], False),
              ("box", 12.2, RY, 3.45, RH, "Use", "clustering | retrieval\nedit a concept, find nearest", "model")])
    return s


STEP_NAMES = ["Title", "DINOv3 teacher and sampler", "Proposer agent", "Merger agent", "Verifier agent",
              "Concept dictionary", "Run time: new micrograph", "Embedding agent", "Concept vector", "Use"]
