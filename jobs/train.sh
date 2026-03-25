#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --gres=gpu:1          # remove if job doesn't need a GPU
#SBATCH --partition=all

set -e

REPO="git@github.com:white-richard/latent-space.git"
BRANCH="${GIT_BRANCH:-main}"
WORKDIR="/tmp/job-${SLURM_JOB_ID}"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "=== Branch: $BRANCH ==="

# 1. Clone the repo at the correct branch/commit
git clone --branch "$BRANCH" "$REPO" "$WORKDIR"
cd "$WORKDIR"
git submodule update --init --recursive

# 2. Set up Python environment
uv venv --python 3.10
fish
source ~/.venv/bin/activate.fish
fish setup.fish --dino

# 3. Pull data via DVC
dvc remote add -d --local wpeb-print /home/richw/.code/latent-space/.dvc/cache
dvc pull /root/.code/latent-space/datasets/cifar.dvc

# 4. Run training — adjust this to your entrypoint
chmod +x src/cifar_lightning/run.fish
./src/cifar_lightning/run.fish --debug-mode

# 5. Push results back via DVC
dvc add out
dvc push

# 6. Clean up
rm -rf "$WORKDIR"

echo "=== Job $SLURM_JOB_ID complete ==="
