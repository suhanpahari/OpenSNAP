import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_DIR = os.environ.get("SNAP_DIR", os.path.join(ROOT, "third_party", "SNAP"))
OPENSCENE_DIR = os.environ.get("OPENSCENE_DIR", os.path.join(ROOT, "third_party", "openscene"))
DOWNLOADS = os.environ.get("OPENSNAP_DOWNLOADS", os.path.join(os.path.dirname(ROOT), "downloads"))
os.environ.setdefault("HF_HOME", os.path.join(DOWNLOADS, "hf"))


def add(path):
    if path not in sys.path:
        sys.path.insert(0, path)
