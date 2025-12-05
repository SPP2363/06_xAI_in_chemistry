# Claude Code Instructions

## Environment Setup

Always activate the virtual environment before running any commands:

```bash
source .venv/bin/activate
```

This ensures packages are installed to the correct environment and not the system or conda environment.

## Package Installation

Prefer using `uv pip` for installing packages:

```bash
uv pip install <package>
```

## Jupyter Notebooks

### Reading Notebooks

**Important**: Do NOT read notebook files (`.ipynb`) directly using the `Read` tool - they contain large binary outputs (images, data) that clutter the context.

Instead, **always strip outputs first** before reading a notebook:

```bash
jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --stdout <notebook.ipynb>
```

This gives you clean source code without execution outputs, images, or other binary data.

### Editing Notebooks

Use the built-in `NotebookEdit` tool to modify notebook cells.

### Executing Notebooks

To execute all cells in a notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace <notebook.ipynb>
```

To start an interactive JupyterLab session:

```bash
python manage.py jupyter-lab
```
