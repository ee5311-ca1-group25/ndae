# NDAE

Pytorch implementation of the paper "Neural Differential Appearance Equations"

## Getting Started

### 1. Install uv

On macOS/Linux:

~~~bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~~~

On Windows (PowerShell):

~~~powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
~~~

### 2. Sync project environment

From the project root, run:

~~~bash
uv sync
~~~

This creates a local virtual environment in .venv and installs all dependencies from pyproject.toml.

### 3. Run the project

~~~bash
uv run python main.py
~~~

### 5. Optional: activate the venv manually

If you prefer an activated shell:

~~~bash
source .venv/bin/activate
~~~

