#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1          # remove if job doesn't need a GPU
#SBATCH --partition=all

set -e

REPO="git@github.com:white-richard/your-repo-name.git"
BRANCH="${GIT_BRANCH:-main}"
WORKDIR="/tmp/job-${SLURM_JOB_ID}"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "=== Branch: $BRANCH ==="

# 1. Clone the repo at the correct branch/commit
git clone --branch "$BRANCH" "$REPO" "$WORKDIR"
cd "$WORKDIR"

# 2. Set up Python environment
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Pull data via DVC
dvc pull

# 4. Run training — adjust this to your entrypoint
python train.py

# 5. Push results back via DVC
dvc push

# 6. Clean up
rm -rf "$WORKDIR"

echo "=== Job $SLURM_JOB_ID complete ==="
