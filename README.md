# Latent Space

Lightly organized machine learning code I've written or gathered, along with a substantial amount of third‑party code. 

## Installation

### How to clone

```bash
git clone --recurse-submodules git@github.com:white-richard/latent-space.git
```

If it is already cloned:

```bash
git submodule update --init --recursive
```

### Installation

Set up a virtual environment (I recommended [`uv`](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.10
source .venv/bin/activate
```

If you're using `fish`, you can run:

```fish
fish setup.fish
```

## License

This project is licensed under the **MIT License** for my original contributions only.

A significant portion of this repository consists of code, snippets, and notebooks sourced from third‑party projects, blogs, papers, and other public resources. For such third‑party material:

- Ownership remains with the original authors.
- The original licenses and terms apply to that code.
- Where possible, attribution is retained in comments, headers, or references.

If you are planning to use any part of this repository beyond personal experimentation or learning, you should:

- Verify the license of the specific files or components you intend to use.
- Attribute the original authors as required by their licenses.
- Remove or replace any code where the license is unclear or incompatible with your use case.

Copyright (c) Richard White (for original code and documentation in this repository)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
