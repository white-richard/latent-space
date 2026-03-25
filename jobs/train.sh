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

set -ex

# Use slurm-jobs' tools, fall back to system paths
export PATH="/home/slurm-jobs/.local/bin:/home/richw/.local/bin:/usr/local/bin:/usr/bin:/bin"

# Force git to use the deploy key regardless of which user is running
export GIT_SSH_COMMAND="ssh -i /home/slurm-jobs/.ssh/github_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

REPO="git@github.com:white-richard/latent-space.git"
BRANCH="${GIT_BRANCH:-main}"
WORKDIR="/tmp/job-${SLURM_JOB_ID}"

echo "=== Job $SLURM_JOB_ID starting on $(hostname) ==="
echo "=== Branch: $BRANCH ==="

git clone --branch "$BRANCH" "$REPO" "$WORKDIR"
cd "$WORKDIR"
git submodule update --init --recursive

# 2. Set up Python environment
uv venv --python 3.10
source .venv/bin/activate
/usr/bin/fish setup.fish --dino

# 3. Pull data via DVC
# Use the local cache on wpeb-print via SSH
# DVC_SSH_KEY is passed in from the workflow env
mkdir -p ~/.ssh
echo "$DVC_SSH_KEY" > ~/.ssh/dvc_key
chmod 600 ~/.ssh/dvc_key
ssh-keyscan wpeb-print >> ~/.ssh/known_hosts 2>/dev/null
dvc remote add -d --local wpeb-print ssh://wpeb-print/home/richw/.code/latent-space/.dvc/cache
dvc remote modify --local wpeb-print keyfile ~/.ssh/dvc_key
dvc pull datasets/cifar.dvc

nvidia-smi

rm -rf out
/usr/bin/fish src/cifar_lightning/run.fish --debug-mode

git config user.email "slurm@wpeb-436-01l"
git config user.name "Slurm Job"
git add out/ 2>/dev/null || true
git diff --cached --quiet || GIT_SSH_COMMAND="ssh -i /home/slurm-jobs/.ssh/github_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" git push origin HEAD

# 6. Clean up
rm -rf "$WORKDIR"

echo "=== Job $SLURM_JOB_ID complete ==="