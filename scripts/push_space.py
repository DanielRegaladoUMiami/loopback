"""Push the Gradio app to a HF Space."""

from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "DanielRegaladoCardoso/loopback"
SPACE_DIR = Path("space")

api = HfApi()
create_repo(REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
print("space repo ready", flush=True)

for f in ["README.md", "requirements.txt", "model.py", "app.py"]:
    print(f"uploading {f}...", flush=True)
    api.upload_file(
        path_or_fileobj=SPACE_DIR / f,
        path_in_repo=f,
        repo_id=REPO_ID,
        repo_type="space",
    )

print(f"\n✅ https://huggingface.co/spaces/{REPO_ID}", flush=True)
