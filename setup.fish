#!/usr/bin/env fish

set -e VIRTUAL_ENV

set latent_dir (dirname (status filename))

set project_dir $PWD

# Parse --dino flag
set use_dino false
for arg in $argv
    if test "$arg" = "--dino"
        set use_dino true
    end
end

# Parse --pe flag
set use_pe false
for arg in $argv
    if test "$arg" = "--pe"
        set use_pe true
    end
end

if test "$use_dino" = true
    uv sync --extra dev --extra dinov3 --project $latent_dir
else
    uv sync --extra dev --project $latent_dir
end

if test "$use_pe" = true
uv pip install -r $latent_dir/repos/pe/requirements.txt
uv pip install -e $latent_dir/repos/pe
end

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r $latent_dir/requirements.txt
uv pip install -e $latent_dir

if test "$use_dino" = true
    uv pip install -e $latent_dir/repos/dinov3
end