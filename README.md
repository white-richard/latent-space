# Latent Space

Lightly organized machine learning code I've written or gathered, along with a substantial amount of third‑party code collected from various sources. This repository serves as a workspace for experiments, utilities, and reusable components related to ML and data science.

> Important: Much of the code in this repository originates from other authors and projects. I do **not** claim ownership of that third‑party code. Where possible, original licenses, copyright notices, and links to sources are preserved or referenced.

## Features

- Collection of ML experimentation scripts and utilities
- Organized structure for reusable components
- Easy local installation for development
- Designed for interactive use in notebooks and scripts

## Installation

### How to clone

Must use --recurse-submodules

```bash
git clone --recurse-submodules git@github.com:white-richard/latent-space.git
```

If it is already cloned:

```bash
git submodule update --init --recursive
```

### Virtual Environment

Set up a virtual environment (recommended to use `uv`):

```bash
uv venv --python 3.10
source .venv/bin/activate
```

### Using `fish` helper script

If you're using `fish`, you can run:

```fish
fish install_dev.fish
```

This is intended to install the project and set up any dependencies.

## Usage

This repository is primarily intended for development and experimentation. Common usage patterns might include:

- Importing utilities into notebooks or scripts
- Running experiment scripts from the command line
- Extending modules with new models or preprocessing steps

Example:

```python
from latent_space.models import SomeModel
from latent_space.data import load_dataset

data = load_dataset("my_dataset")
model = SomeModel()
model.fit(data)
```

## Development

If you want to hack on this project:

1. **Create and activate a virtual environment**

    ```bash
    uv venv --python 3.10
    source .venv/bin/activate
    ```

2. **Install development dependencies**

    Using `uv`:

    ```bash
    uv pip install -e ".[dev]"
    ```

    Or using the provided `fish` script:

    ```fish
    fish install_dev.fish
    ```

3. **Run tests** (coming soon...)

    ```bash
    pytest
    ```

## Contributing

Contributions, ideas, and suggestions are welcome. If you plan to contribute:

1. Fork the repository on GitHub.
2. Create a feature branch:

    ```bash
    git checkout -b feature/my-change
    ```

3. Make your changes with clear, focused commits.
4. Add or update tests as appropriate.
5. Open a pull request against the main branch, describing:
    - What you changed
    - Why you changed it
    - Any relevant context or trade-off

# Dataset Management (DVC)

This repo uses [DVC](https://dvc.org/) to version and sync large assets (datasets, model weights, feature points) across machines. Actual files are **not stored in Git** — only lightweight `.dvc` pointer files are committed. The real data lives on my remote SSH servers (not accessible publically).

## First-time setup on a new machine

### Install DVC

```bash
pip install dvc dvc-ssh
```

### Configure your remotes

```bash
# Add each remote server
dvc remote add --local your-server ssh://your-server/path/to/latent-space/datasets

# Set your SSH key for each remote
dvc remote modify --local your-server keyfile ~/.ssh/id_ed25519
# dvc remote modify --local your-server keyfile ~/.ssh/id_ed25519

# Set the default remote
dvc remote default --local your-server
```

## Pulling datasets

Pull everything from the default remote:

```bash
dvc pull
```

Pull a specific dataset only:

```bash
dvc pull datasets/dataset1.dvc
```

Pull from a specific remote:

```bash
dvc pull datasets/dataset1.dvc -r your-server
```

## Adding a new dataset

```bash
# Track it with DVC
dvc add datasets/new-dataset/

# Commit the pointer file to Git
git add datasets/new-dataset.dvc .gitignore
git commit -m "track new-dataset with dvc"

# Push the actual files to the remote
dvc push datasets/new-dataset.dvc -r your-server
```

## Pushing updated files

```bash
dvc add datasets/updated-dataset/
git add datasets/updated-dataset.dvc
git commit -m "update dataset: describe what changed"
dvc push datasets/updated-dataset.dvc -r your-server
```

## Checking sync status

See what's out of sync between your local machine and the remote:

```bash
dvc status -c
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
