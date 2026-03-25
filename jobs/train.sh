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
  -e GITHUB_DEPLOY_KEY="$(cat /home/richw/.ssh/github_deploy)" \
  -e DVC_SSH_KEY="$DVC_SSH_KEY" \
  -e GIT_BRANCH="${GIT_BRANCH:-main}" \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" \
  ml-runner:latest bash -c '
    # Write keys
    mkdir -p ~/.ssh
    echo "$GITHUB_DEPLOY_KEY" > ~/.ssh/github_deploy
    chmod 600 ~/.ssh/github_deploy
    export GIT_SSH_COMMAND="ssh -i ~/.ssh/github_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    # Clone
    git clone --branch "$GIT_BRANCH" git@github.com:white-richard/latent-space.git /workspace
    cd /workspace
    git submodule update --init --recursive

    # Setup
    fish setup.fish --dino

    # DVC
    echo "$DVC_SSH_KEY" > ~/.ssh/dvc_key
    chmod 600 ~/.ssh/dvc_key
    dvc remote add -d --local wpeb-print ssh://wpeb-print/home/richw/.code/latent-space/.dvc/cache
    dvc remote modify --local wpeb-print keyfile ~/.ssh/dvc_key
    ssh-keyscan wpeb-print >> ~/.ssh/known_hosts 2>/dev/null
    dvc pull datasets/cifar.dvc

    nvidia-smi

    # Train
    fish src/cifar_lightning/run.fish --debug-mode

    # Push results
    git config user.email "slurm@cluster"
    git config user.name "Slurm Job"
    git add out/ 2>/dev/null || true
    git diff --cached --quiet || git push origin HEAD
  '

echo "=== Job $SLURM_JOB_ID complete ==="