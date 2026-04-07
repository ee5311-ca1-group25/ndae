# NDAE

PyTorch implementation of the paper "Neural Differential Appearance Equations".

The repository includes config loading, exemplar sampling, differentiable
svBRDF rendering, perceptual/statistical losses, a minimal training runtime,
checkpoint save-resume flow, and checkpoint-based sampling.

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

### 3. Run a Dry Run

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

### 4. Train on the Full `clay_solidifying` Sequence

The full-sequence config uses all 100 frames listed in
`data_local/svbrdf_full/clay_solidifying/_manifest.json`.

~~~bash
uv run python scripts/train_svbrdf.py --config configs/full_clay.yaml
~~~

After training, sample from the latest refresh-boundary checkpoint:

~~~bash
uv run python scripts/sample_svbrdf.py \
  --checkpoint outputs/full_clay_solidifying/checkpoints/latest
~~~

Render a static loss plot from the recorded metrics:

~~~bash
uv run python scripts/plot_metrics.py \
  outputs/full_clay_solidifying/metrics.jsonl \
  --output outputs/full_clay_solidifying/loss_curve.png
~~~

### 5. Download a Mini SVBRDF Subset

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

### 6. Run Tests

~~~bash
uv run pytest
~~~

For the current rendering-helper slice, the narrow regression command is:

~~~bash
uv run pytest tests/test_renderer.py tests/test_package_layout.py tests/test_config.py -q
~~~

### 7. Render a Synthetic svBRDF Example

~~~bash
uv run python scripts/render_svbrdf_example.py \
  --preset plastic \
  --output outputs/render_example/plastic.png \
  --image-size 256
~~~

Or render the darker coated-metal preset:

~~~bash
uv run python scripts/render_svbrdf_example.py \
  --preset coated_metal \
  --output outputs/render_example/coated_metal.png \
  --image-size 256
~~~

These presets use smoother, more physically intuitive BRDF maps so the
highlights look closer to painted plastic and coated metal than the earlier
debug-pattern example.

### 8. Optional: activate the venv manually

If you prefer an activated shell:

~~~bash
source .venv/bin/activate
~~~
