#!/usr/bin/env fish
set -e VIRTUAL_ENV

set latent_dir (dirname (status filename))
set project_dir $PWD

# dvc gc --workspace

git submodule update --remote --recursive
git -C $latent_dir pull --recurse-submodules

uv sync --project $latent_dir $args
