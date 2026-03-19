#!/usr/bin/env fish

set -e VIRTUAL_ENV

set latent_dir "$HOME/.code/latent-space/"

set project_dir $PWD

# Parse --dino flag
set use_dino false
for arg in $argv
    if test "$arg" = "--dino"
        set use_dino true
    end
end

uv pip install -r $latent_dir/requirements.txt
uv sync --extra dev --project $latent_dir
uv pip install -e $latent_dir

if test "$use_dino" = true
    uv pip install -e $latent_dir/repos/dinov3
    uv sync --extra dinov3 --project $latent_dir
end
