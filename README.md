# Setup

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create the virtual environment

```bash
uv venv --python 3.13
```

## Install dependencies

```bash
uv pip install -e .
```

## Register the kernel with Jupyter

```bash
uv run python -m ipykernel install --user --name my-project --display-name "☕ Demo (3.13)"
```

The environment will now appear as **my-project (3.13)** in JupyterLab / Jupyter Notebook.

## Removing the kernel

```bash
jupyter kernelspec remove my-project
```
