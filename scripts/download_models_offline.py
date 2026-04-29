"""
Download all models used by run_sequential_training_run2 to a local directory.
Run once on a machine with internet (e.g. login node): python scripts/download_models_offline.py
Then set LOCAL_MODELS_ROOT (default: project_root/pretrained_models) so SLURM jobs load from disk.
"""
import os
import sys

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huggingface_hub import snapshot_download

# Same models as run_sequential_training_run2.sh (repo_id -> local path = pretrained_models/repo_id)
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Download models for offline use on compute nodes.")
    p.add_argument("--output-dir", "-o", default=os.path.join(PROJECT_ROOT, "pretrained_models"),
                   help="Base directory (default: PROJECT_ROOT/pretrained_models)")
    p.add_argument("--token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
                   help="HF token for gated models (or set HUGGINGFACE_HUB_TOKEN)")
    args = p.parse_args()
    base = os.path.abspath(args.output_dir)
    os.makedirs(base, exist_ok=True)
    print(f"Downloading to {base}")
    token = args.token
    for repo_id in MODELS:
        local_dir = os.path.join(base, repo_id.replace("/", os.sep))
        if os.path.isdir(local_dir) and any(f.startswith("config") for f in os.listdir(local_dir)):
            print(f"Skip (exists): {repo_id}")
            continue
        print(f"Downloading: {repo_id} -> {local_dir}")
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token)
        except Exception as e:
            print(f"ERROR {repo_id}: {e}")
            raise
    print("Done. Set LOCAL_MODELS_ROOT=%s (or export before sbatch) and run training." % base)


if __name__ == "__main__":
    main()
