---
title: OpenSNAP
emoji: 🧊
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Phone video → 3D scene → open-vocabulary object queries
suggested_hardware: zero-a10g
models:
  - suhanxvii/opensnap-weights
  - facebook/VGGT-1B
  - depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
  - Qwen/Qwen2.5-VL-7B-Instruct
---

# OpenSNAP

Upload a short phone video of a static scene (a desk, a shelf, a room). OpenSNAP

1. reconstructs it in 3D with **VGGT**, sets metric scale with **Depth-Anything-V2 (metric)** and aligns it to gravity;
2. segments it into objects with **SNAP** (promptable point-cloud segmentation);
3. lets you **click an object in any video frame** — the click is lifted to 3D, the object is segmented (positive and
   *negative* clicks, test-time token tuning) and named by **Qwen2.5-VL** from the real photo;
4. answers **text queries** — a word (`monitor`) or OVQL (`SELECT WHERE IS('chair') AND NEAR(IS('desk'), 0.5)`).

Code, paper and benchmarks: <https://github.com/suhanpahari/OpenSNAP>

Licences: SNAP (MIT); VGGT-1B and Depth-Anything-V2-Metric-Large (CC BY-NC 4.0) — non-commercial demo.
