#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --partition=all

set -ex   # -x prints every command before executing it

export PATH="/home/richw/.local/bin:$PATH"

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
# sudo chown -R richw:richw ~/.cache/uv
# chmod -R 777 ~/.cache/uv
uv venv --python 3.10
source .venv/bin/activate
/usr/bin/fish setup.fish --dino

# 3. Pull data via DVC
dvc remote add -d --local wpeb-print /home/richw/.code/latent-space/.dvc/cache
dvc pull /home/richw/.code/latent-space/datasets/cifar.dvc

nvidia-smi

rm -rf out

# 4. Run training — adjust this to your entrypoint
fish src/cifar_lightning/run.fish --debug-mode

# 5. Push results back via DVC
# dvc add out
# dvc push -r wpeb-print
git wave "good [skip ci]"

# 6. Clean up
rm -rf "$WORKDIR"

echo "=== Job $SLURM_JOB_ID complete ==="
