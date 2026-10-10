"""Stage and publish the Space (needs an HF token with write access in HF_TOKEN and a PRO account for Gradio + ZeroGPU)."""
import os
import subprocess
import sys

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
SPACE = os.environ.get("OPENSNAP_SPACE", "suhanxvii/OpenSNAP")
stage = os.path.join(os.path.dirname(os.path.dirname(HERE)), "downloads", "space_stage")
subprocess.run([sys.executable, os.path.join(HERE, "stage.py"), stage], check=True)
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(folder_path=stage, repo_id=SPACE, repo_type="space", commit_message="OpenSNAP demo")
try:
    api.request_space_hardware(SPACE, "zero-a10g")
except Exception as e:
    print("could not request ZeroGPU (choose hardware in the Space settings):", e)
print("https://huggingface.co/spaces/" + SPACE)
