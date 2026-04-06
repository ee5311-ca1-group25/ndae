# NDAE

Pytorch implementation of the paper "Neural Differential Appearance Equations"

This repository is being built incrementally as a course project. The current
state corresponds to Lecture 3 Phase D: package layout, config loading,
workspace creation, rendering metadata, latent-map extraction helpers,
height-to-normal conversion, and the core differentiable svBRDF renderer are in
place. The rendering core is now split across `rendering/geometry.py`,
`rendering/brdf.py`, and `rendering/renderer.py`; the full training pipeline is
not implemented yet.

## Documentation

- GitHub Pages: https://ee5311-ca1-group25.github.io/ndae_doc/
- In the pydae superproject, docs source is mounted as the `ndae-docs` submodule.

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

### 3. Run the Lecture 1 Dry Run

~~~bash
uv run python main.py --config configs/base.yaml --dry-run
~~~

You can also use the script entry point:

~~~bash
uv run python scripts/train_svbrdf.py --config configs/base.yaml --dry-run
~~~

The dry run will:

- load the config,
- create `outputs/<experiment.name>/`,
- write `config.resolved.yaml`,
- print a run summary,
- exit without training.

### 4. Download a Mini SVBRDF Subset

This project includes a helper script that samples a tiny local subset from the
official `SVBRDF_dynamic_flash_textures.zip` without downloading the full 11.96GB
archive.

Prerequisite: `npx` must be available on your machine.

~~~bash
uv run python scripts/download_svbrdf_mini.py --exemplar clay_solidifying --count 4
~~~

The files are written under `data_local/svbrdf_mini/<exemplar>/` and stay out of
git because `data_local/` is ignored.

If the UCL site blocks the automated browser and the script reports `403 Forbidden`,
you can reuse the same script with a manual fallback:

~~~bash
uv run python scripts/download_svbrdf_mini.py \
  --exemplar clay_solidifying \
  --count 4 \
  --cookie-header 'aws-waf-token=...; FIGINSTWEBIDCD=...'
~~~

Or, if you have already copied the redirected S3 ZIP URL from your browser's
network panel:

~~~bash
uv run python scripts/download_svbrdf_mini.py \
  --exemplar clay_solidifying \
  --count 4 \
  --signed-url 'https://s3-eu-west-1.amazonaws.com/.../SVBRDF_dynamic_flash_textures.zip?...'
~~~

There is also a semi-automatic mode that opens your normal browser and then
waits for you to paste the request `Cookie` header or redirected S3 ZIP URL:

~~~bash
uv run python scripts/download_svbrdf_mini.py \
  --exemplar clay_solidifying \
  --count 8 \
  --semi-auto
~~~

This is the preferred fallback when the repository site blocks Playwright with
`403 Forbidden`.

### 5. Run Tests

~~~bash
uv run pytest
~~~

For the current rendering-helper slice, the narrow regression command is:

~~~bash
uv run pytest tests/test_renderer.py tests/test_package_layout.py tests/test_config.py -q
~~~

### 6. Optional: activate the venv manually

If you prefer an activated shell:

~~~bash
source .venv/bin/activate
~~~
