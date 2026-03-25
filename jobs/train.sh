#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --partition=all
#SBATCH --account=
#SBATCH --exclusive

set -ex


# ============================================================
# PROJECT-SPECIFIC CONFIGURATION
# ============================================================
GITHUB_REPO="white-richard/latent-space"
DATA_MOUNT="/home/richw/.code/latent-space"   # local data cache path on each machine
RUN_SCRIPT="src/cifar_lightning/run.fish"
SETUP_CMD="fish setup.fish --dino"
DVC_PULL_PATH="datasets/cifar.dvc"
GIT_USER_EMAIL="98299003+white-richard@users.noreply.github.com"
GIT_USER_NAME="white-richard"
# ============================================================

BRANCH="${GIT_BRANCH:-main}"

docker run --rm --device nvidia.com/gpu=all \
  -e GIT_BRANCH="$BRANCH" \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" \
  -e GITHUB_REPO="$GITHUB_REPO" \
  -e DEFAULT_RUN_SCRIPT="$DEFAULT_RUN_SCRIPT" \
  -e RUN_SCRIPT="$RUN_SCRIPT" \
  -e SETUP_CMD="$SETUP_CMD" \
  -e DVC_PULL_PATH="$DVC_PULL_PATH" \
  -e GIT_USER_EMAIL="$GIT_USER_EMAIL" \
  -e GIT_USER_NAME="$GIT_USER_NAME" \
  -v /home/slurm-jobs/.ssh/github_deploy:/root/.ssh/github_deploy:ro \
  -v /home/slurm-jobs/.ssh/dvc_key:/root/.ssh/dvc_key:ro \
  -v "${DATA_MOUNT}:/root/.code/data:ro" \
  -v "${HOME}/.cache/uv:/root/.cache/uv" \
  ml-runner:latest bash -c '
    export GIT_SSH_COMMAND="ssh -i /root/.ssh/github_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    git clone --branch "$GIT_BRANCH" git@github.com:${GITHUB_REPO}.git /workspace
    cd /workspace
    git submodule update --init --recursive
    eval "$SETUP_CMD"
    source .venv/bin/activate
    dvc remote add -d --local local-cache /root/.code/data/.dvc/cache
    dvc pull "$DATA_MOUNT/$DVC_PULL_PATH"
    nvidia-smi
    # ── run training ─────────────────────────────────────────
    [ -f "$RUN_SCRIPT" ] || RUN_SCRIPT="$DEFAULT_RUN_SCRIPT"
    echo "=== Running: $RUN_SCRIPT ==="
    fish "$RUN_SCRIPT" --debug-mode
    # ── push results ─────────────────────────────────────────
    git config user.email "$GIT_USER_EMAIL"
    git config user.name "$GIT_USER_NAME"
    git add out/ 2>/dev/null || true
    git diff --cached --quiet || git push origin HEAD
  '

echo "=== Job $SLURM_JOB_ID complete ==="