#!/usr/bin/env fish

set -l repo_root (pwd)
set -l venv_python "$repo_root/.venv/bin/python"

if not test -x "$venv_python"
    echo "Error: '$venv_python' not found or not executable."
    echo "Create it with: uv venv"
    exit 1
end

# Editable install should make latent_space importable; no PYTHONPATH needed
exec "$venv_python" -m latent_space.lightning.run_experiments $argv
