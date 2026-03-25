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


docker run --rm --gpus all \
  -e GIT_BRANCH="${GIT_BRANCH:-main}" \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" \
  -v /home/slurm-jobs/.ssh/github_deploy:/root/.ssh/github_deploy:ro \
  -v /home/slurm-jobs/.ssh/dvc_key:/root/.ssh/dvc_key:ro \
  -v /home/richw/.code/latent-space:/root/.code/latent-space:ro \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  -v "${HOME}/.cache/uv:/root/.cache/uv" \
  ml-runner:latest bash -c '
    chmod 600 /root/.ssh/github_deploy /root/.ssh/dvc_key
    export GIT_SSH_COMMAND="ssh -i /root/.ssh/github_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    git clone --branch "$GIT_BRANCH" git@github.com:white-richard/latent-space.git /workspace
    cd /workspace
    git submodule update --init --recursive
    fish setup.fish --dino
    source .venv/bin/activate
    dvc remote add -d --local local-cache /root/.code/latent-space/.dvc/cache
    dvc pull /root/.code/latent-space/datasets/cifar.dvc
    nvidia-smi
    RUN_SCRIPT="${RUN_SCRIPT:-src/cifar_lightning/run.fish}"
    fish "$RUN_SCRIPT" --debug-mode
    git config user.email "98299003+white-richard@users.noreply.github.com"
    git config user.name "white-richard"
    git add out/ 2>/dev/null || true
    git diff --cached --quiet || git push origin HEAD
  '

echo "=== Job $SLURM_JOB_ID complete ==="