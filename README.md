# Explainable AI for Molecules

This repository contains tutorial notebooks for applying Explainable AI (XAI) methods to molecular property prediction. The tutorials serve as a practical companion to our review paper, providing hands-on examples of various explanation methods.

## Setup & Installation

This section provides instructions for setting up the tutorial environment on your local machine.

### Requirements

- **Python 3.9+** (tested with Python 3.9, 3.10, 3.11, and 3.12)
- **uv** — a fast Python package manager (required for installing dependencies from git sources)
- **Git** — for cloning the repository

GPU acceleration via CUDA is supported and recommended for faster model training, but not strictly required. The tutorials will run on CPU if no GPU is available.

### Linux

**1. Install uv** (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repository:**

```bash
git clone https://github.com/aimat-lab/xai_chem_review.git
cd xai_chem_review
```

**3. Create a virtual environment and install dependencies:**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

**4. Verify the installation:**

```bash
python -c "import xai_chem_review; import torch; import rdkit; print('Installation successful!')"
```

**5. Launch Jupyter:**

```bash
jupyter lab
```

### Windows

**1. Install uv** (if not already installed):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Clone the repository:**

```powershell
git clone https://github.com/aimat-lab/xai_chem_review.git
cd xai_chem_review
```

**3. Create a virtual environment and install dependencies:**

```powershell
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
```

**4. Verify the installation:**

```powershell
python -c "import xai_chem_review; import torch; import rdkit; print('Installation successful!')"
```

**5. Launch Jupyter:**

```powershell
jupyter lab
```

> **Note:** Some dependencies may have limited support on Windows. If you encounter issues, consider using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) and following the Linux instructions instead.

## Changelog

### 0.1.0 - Initial version

Initial release of the tutorial notebooks.
