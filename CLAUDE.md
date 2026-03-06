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

### Reading and Editing Notebooks

Always use the MCP server tools for reading and editing Jupyter notebooks:

- **Reading notebooks**: Use `mcp__jupyter-notebook__read_notebook_source_only` to read notebook content. This returns cell sources without outputs, which is more efficient for understanding notebook structure.
- **Reading with outputs**: Use `mcp__jupyter-notebook__read_notebook_with_outputs` only when you need to see cell execution outputs.
- **Reading specific cell output**: Use `mcp__jupyter-notebook__read_output_of_cell` to read the output of a specific cell by its ID.
- **Editing cells**: Use `mcp__jupyter-notebook__edit_cell` to modify existing cell content.
- **Adding cells**: Use `mcp__jupyter-notebook__add_cell` to insert new cells at a specific position.
- **Executing cells**: Use `mcp__jupyter-notebook__execute_cell` to run a specific cell and retrieve its output.

Do NOT use the standard `Read`, `Edit`, or `Write` tools for `.ipynb` files, as these tools do not properly handle the JSON structure of Jupyter notebooks.

### Executing Notebooks

To execute all cells in a notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace <notebook.ipynb>
```

To start an interactive JupyterLab session:

```bash
python manage.py jupyter-lab
```
