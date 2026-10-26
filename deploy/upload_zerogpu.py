"""Create (or update) the free ZeroGPU Space from deploy/zerogpu/.

Usage:
    python deploy/upload_zerogpu.py
Reads HF_TOKEN from .env. Free personal accounts can host up to 2 ZeroGPU
Spaces once the account is 30+ days old with a verified email; before that,
Hugging Face answers 402 Payment Required.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

api = HfApi(token=os.environ["HF_TOKEN"])
SPACE_REPO = f"{api.whoami()['name']}/medical-note-summarizer"


def main() -> None:
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio",
                    space_hardware="zero-a10g", exist_ok=True)
    api.upload_folder(folder_path=ROOT / "deploy" / "zerogpu", repo_id=SPACE_REPO,
                      repo_type="space")
    print(f"Space: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
