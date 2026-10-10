"""Assemble the Hugging Face Space folder (app + the parts of opensnap and SNAP it needs)."""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(ROOT), "downloads", "space_stage")
PKG = ["__init__.py", "paths.py", "core.py", "snap_model.py", "snap_plus.py", "clicks.py", "namer.py", "photo.py", "render.py",
       "text.py", "data.py", "video.py", "lora.py", "query/__init__.py", "query/lang.py", "query/catalog.py", "query/engine.py",
       "vocab/lvis.txt", "vocab/snap_train.txt"]
SNAP = ["LICENSE", "src", "datasets/transforms.py"]

if os.path.exists(OUT):
    shutil.rmtree(OUT)
for f in ("app.py", "requirements.txt", "README.md"):
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "space", f), os.path.join(OUT, f))
for f in PKG:
    dst = os.path.join(OUT, "opensnap", f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "opensnap", f), dst)
for f in SNAP:
    src, dst = os.path.join(ROOT, "third_party", "SNAP", f), os.path.join(OUT, "third_party", "SNAP", f)
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
open(os.path.join(OUT, "third_party", "SNAP", "datasets", "__init__.py"), "w").write("")
os.makedirs(os.path.join(OUT, "examples"), exist_ok=True)
shutil.copy(os.path.join(ROOT, "assets", "demo_lab_desk", "lab_desk.mp4"), os.path.join(OUT, "examples", "lab_desk.mp4"))
print("staged", OUT)
