#!/usr/bin/env fish

uv pip install -r requirements.txt
uv sync --extra dev
# check for --dino boolean flag, if so, then
uv pip install -e $PWD/repo/dinov3
uv sync --extra dinov3