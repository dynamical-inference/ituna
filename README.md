# 🐟iTuna

**Tune machine learning models for empirical identifiability and consistency**

[![PyPI version](https://img.shields.io/pypi/v/ituna.svg)](https://pypi.org/project/ituna/)
[![Python versions](https://img.shields.io/pypi/pyversions/ituna.svg)](https://pypi.org/project/ituna/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/dynamical-inference/ituna/actions/workflows/pytest.yml/badge.svg)](https://github.com/dynamical-inference/ituna/actions/workflows/pytest.yml)

## Why 🐟iTuna?

Applying machine learning to scientific data analysis often suffers from an **identifiability gap**: many models along the data-to-analysis pipeline lack statistical guarantees about the uniqueness of their learned representations. This means that re-running the same algorithm can yield different embeddings, making downstream interpretation unreliable without manual verification.

Identifiable representation learning addresses this by ensuring models recover representations that are unique up to a known class of transformations (permutation, linear, affine, etc.). However, even theoretically identifiable models need **empirical validation** to confirm they behave consistently in practice.

🐟iTuna closes this gap by providing a lightweight, model-agnostic framework to:

1. Train multiple instances of a model with different random seeds
2. Align their embeddings under the appropriate indeterminacy class
3. Measure how consistent the learned representations are

Think of it as a **unit test for reproducibility** of learned embeddings.

## Features

- **sklearn-compatible**: Works with any transformer implementing `fit`, `transform`, and standard sklearn conventions
- **Built-in indeterminacy classes**:
  - `Identity` - no transformation needed (model is already fully identifiable)
  - `Permutation` - handles sign flips and component reordering (e.g., FastICA)
  - `Linear` - linear transformation alignment (e.g., PCA)
  - `Affine` - linear transformation with intercept (e.g., CEBRA)
- **Consistency scoring**: Quantifies how stable embeddings are across runs
- **Embedding alignment**: Returns aligned embeddings for downstream analysis
- **Flexible backends**: In-memory, disk caching, distributed execution, and DataJoint support

## Installation

```bash
pip install ituna
```

Optional extras:

```bash
pip install ituna[datajoint]  # DataJoint backend for database-backed caching
pip install ituna[dev]        # Development dependencies (pytest, etc.)
```

## Quickstart

```python
import numpy as np
from sklearn.decomposition import FastICA

from ituna import ConsistencyEnsemble, metrics

# Generate sample data
X = np.random.randn(1000, 64)

# Create a consistency ensemble
ensemble = ConsistencyEnsemble(
    estimator=FastICA(n_components=16, max_iter=500),
    consistency_transform=metrics.PairwiseConsistency(
        indeterminacy=metrics.Permutation(),  # FastICA is identifiable up to permutation
        symmetric=False,
        include_diagonal=True,
    ),
    random_states=5,  # Train 5 instances with different seeds
)

# Fit and evaluate
ensemble.fit(X)
print("Consistency score:", ensemble.score(X))

# Get aligned embeddings
emb = ensemble.transform(X)
print("Embedding shape:", emb.shape)
```

## Documentation

- **Quickstart notebook**: [`docs/tutorials/quickstart.ipynb`](docs/tutorials/quickstart.ipynb) - minimal working example
- **Reference walkthrough**: [`iTune Reference.ipynb`](iTune%20Reference.ipynb) - comprehensive tutorial
- **Real-world examples**: [`ituna-experiments/`](ituna-experiments/) - experiments with CEBRA and other models

## Backends

🐟iTuna supports different backends for caching and distributed computation:

```python
from ituna import ConsistencyEnsemble, config, metrics
from sklearn.decomposition import FastICA

ensemble = ConsistencyEnsemble(
    estimator=FastICA(n_components=16, max_iter=500),
    consistency_transform=metrics.PairwiseConsistency(
        indeterminacy=metrics.Permutation(),
    ),
    random_states=10,
)

# Enable disk caching (avoids re-fitting identical models)
with config.config_context(DEFAULT_BACKEND="disk_cache"):
    ensemble.fit(X)

# Distributed execution with multiple workers
with config.config_context(
    DEFAULT_BACKEND="disk_cache_distributed",
    BACKEND_KWARGS={"trigger_type": "auto", "num_workers": 4},
):
    ensemble.fit(X)
```

### CLI Commands

For large-scale experiments, use the command-line tools:

```bash
# Local distributed backend
ituna-fit-distributed --sweep-name <sweep-uuid> --cache-dir ./cache

# DataJoint backend
ituna-fit-distributed-datajoint --sweep-name <sweep-uuid> --schema-name myschema
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/dynamical-inference/ituna.git
cd ituna
pip install -e .[dev]

# Run tests
pytest tests -v

# Setup pre-commit hooks
pre-commit install
```

### Building Documentation

The documentation is built using [Jupyter Book](https://jupyterbook.org/). Note that version 2.x has a completely different build system, so we require version 1.x.

#### Local Build

```bash
# Install jupyter-book (must be version <2)
pip install "jupyter-book<2"

# Build the docs from the project root
jupyter-book build .

# The HTML output will be in _build/html/
# Open in browser:
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

#### Local Server

To serve the docs locally with live preview:

```bash
# After building, start a local server
cd _build/html
python -m http.server 8080

# Then open http://localhost:8080 in your browser
```

#### Using Docker (Recommended)

For a consistent build environment with auto-rebuild on file changes, use the provided Docker setup:

```bash
# Build and run the docs server (includes watchexec for auto-rebuild)
./build.sh

# This will:
# 1. Build the Docker image with all dependencies
# 2. Mount the current directory into the container
# 3. Build the docs and start a server at http://localhost:8000
# 4. Watch for changes and automatically rebuild

# Press Ctrl+C to stop
```

The Docker setup uses `entrypoint.sh` which:
- Builds the Jupyter Book
- Starts a local server on port 8000
- Watches for changes to `.ipynb`, `.md`, `.yml`, and `.py` files and rebuilds automatically

## Citation

If you use 🐟iTuna in your research, please cite:

```bibtex
@software{ituna,
  author = {Schmidt, Tobias and Schneider, Steffen},
  title = {iTuna: Tune machine learning models for empirical identifiability and consistency},
  url = {https://github.com/dynamical-inference/ituna},
  version = {0.3.2},
}
```

## License

🐟iTuna is released under the [MIT License](https://opensource.org/licenses/MIT).
