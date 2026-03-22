#!/usr/bin/env fish

set -l script_dir (dirname (status --current-filename))
set -l repo_root (realpath "$script_dir/../..")
set -l venv_python "$repo_root/.venv/bin/python"

if not test -x "$venv_python"
    echo "Error: '$venv_python' not found or not executable."
    echo "Create it with: uv venv"
    exit 1
end

cd "$repo_root"
$venv_python -m src.cifar_lightning.experiments $argv
