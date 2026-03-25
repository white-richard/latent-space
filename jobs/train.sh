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

docker run --rm --gpus all \
  -e GIT_BRANCH="${GIT_BRANCH:-main}" \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" \
  -v /home/slurm-jobs/.ssh/github_deploy:/root/.ssh/github_deploy:ro \
  -v /home/slurm-jobs/.ssh/dvc_key:/root/.ssh/dvc_key:ro \
  -v "${HOME}/.cache/uv:/root/.cache/uv" \
  ml-runner:latest bash -c '
    chmod 600 /root/.ssh/github_deploy /root/.ssh/dvc_key
    export GIT_SSH_COMMAND="ssh -i /root/.ssh/github_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    git clone --branch "$GIT_BRANCH" git@github.com:white-richard/latent-space.git /workspace
    cd /workspace
    git submodule update --init --recursive
    fish setup.fish --dino
    source .venv/bin/activate
    dvc remote add -d --local wpeb-print ssh://wpeb-print/home/richw/.code/latent-space/.dvc/cache
    dvc remote modify --local wpeb-print keyfile /root/.ssh/dvc_key
    ssh-keyscan wpeb-print >> ~/.ssh/known_hosts 2>/dev/null
    dvc pull datasets/cifar.dvc
    nvidia-smi
    fish src/cifar_lightning/run.fish --debug-mode
    git config user.email "slurm@local"
    git config user.name "Richard White"
    git add out/ 2>/dev/null || true
    git diff --cached --quiet || git push origin HEAD
  '

echo "=== Job $SLURM_JOB_ID complete ==="