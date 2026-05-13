#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ankita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=98299003+white-richard@users.noreply.github.com

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4

# ==========================================
# Parse Command Line Arguments
# ==========================================

# The first argument passed to the script is the project directory
PROJECT_DIR=$1
shift

if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: No project directory provided."
    echo "Usage: sbatch job.sh <path_to_project> <python_script.py> [args...]"
    exit 1
fi

# ==========================================
# Setup Environment
# ==========================================

cd "$PROJECT_DIR" || exit 1
mkdir -p logs

fish setup.fish
source .venv/bin/activate

# ==========================================
# Run Code
# ==========================================

SCRIPT=$1
case "${SCRIPT##*.}" in
    py)   INTERP="python" ;;
    sh)   INTERP="sh" ;;
    bash) INTERP="bash" ;;
    fish) INTERP="fish" ;;
    *)    echo "ERROR: Unknown script type '${SCRIPT##*.}'. Supported: .py, .sh, .bash, .fish"
          exit 1 ;;
esac

echo "=========================================="
echo "Executing: $INTERP $@"
echo "=========================================="

$INTERP "$@"
