import time

import numpy as np
from sklearn.decomposition import FastICA

import ituna


def generate_ica_data(n_samples=2000, n_sources=7):
    """Create ICA-style data with mixed independent sources."""
    t = np.linspace(0, 10, n_samples)
    sources = np.column_stack([np.sin(i * t) for i in range(1, n_sources + 1)])
    mixing_matrix = np.random.randn(n_sources, n_sources)
    X = sources @ mixing_matrix.T
    X += 0.3 * np.random.randn(*X.shape)
    return X


def run_ica_experiment(estimator, data, random_states=[0, 1, 2]):
    """Runs a simple ICA experiment with a ConsistencyEnsemble."""
    print(f"\n--- Running experiment with model: {estimator} ---")
    ensemble = ituna.ConsistencyEnsemble(
        estimator=estimator,
        consistency_transform=ituna.metrics.PairwiseConsistency(
            indeterminacy=ituna.metrics.Permutation(),
            symmetric=False,
            include_diagonal=True,
        ),
        random_states=random_states,
    )

    start_time = time.time()
    ensemble.fit(data)
    end_time = time.time()

    score = ensemble.score(data)
    print(f"Ensemble score: {score:.4f}")
    print(f"Fitting took: {end_time - start_time:.2f} seconds")
    return score


def example_in_memory(data):
    """Demonstrates the default in-memory backend (no context)."""
    print("=" * 50)
    print("### Example 0: Default In-Memory Backend (No Caching) ###")
    print("=" * 50)

    # No config_context is used, so ituna falls back to the default in-memory backend.
    # Caching is not performed between runs.
    model = FastICA(n_components=7, max_iter=2000)

    # First run: models are fitted in memory.
    print("\nFirst run (fitting in memory):")
    run_ica_experiment(model, data)

    # Second run: models are fitted in memory again.
    # Note: The fitting time will be similar to the first run as there is no caching.
    print("\nSecond run (fitting in memory again):")
    run_ica_experiment(model, data)


def example_disk_cache(data):
    """Demonstrates the standard disk_cache backend."""
    print("\n" + "=" * 50)
    print("### Example 1: Standard Disk Cache Backend ###")
    print("=" * 50)
    print("Using default cache directory: ./cache")

    with ituna.config.config_context(DEFAULT_BACKEND="disk_cache"):
        model = FastICA(n_components=7, max_iter=2001)

        # First run: models are fitted and cached.
        print("\nFirst run (fitting and caching):")
        run_ica_experiment(model, data)

        # Second run: models are loaded from cache, should be much faster.
        print("\nSecond run (loading from cache):")
        run_ica_experiment(model, data)


def example_distributed_multiprocess(data):
    """Demonstrates the distributed backend with multi-process trigger."""
    print("\n" + "=" * 50)
    print("### Example 2: Distributed Backend (multi-process) ###")
    print("=" * 50)
    print("Using default cache directory: ./cache")

    backend_kwargs = {
        "trigger_type": "auto",
        "num_workers": 4,
    }

    with ituna.config.config_context(
        DEFAULT_BACKEND="disk_cache_distributed",
        BACKEND_KWARGS=backend_kwargs,
    ):
        # Using a different n_components to avoid cache hits from previous examples
        model = FastICA(n_components=7, max_iter=2002)

        print("\nFitting models with 4 worker processes:")
        run_ica_experiment(model, data)


def example_distributed_manual(data):
    """Demonstrates the distributed backend with manual trigger."""
    print("\n" + "=" * 50)
    print("### Example 3: Distributed Backend (manual trigger) ###")
    print("=" * 50)
    print("Using default cache directory: ./cache")

    backend_kwargs = {
        "trigger_type": "manual",
        "sweep_type": "constant",
        "sweep_name": "manual_ica_experiment",
    }

    with ituna.config.config_context(
        DEFAULT_BACKEND="disk_cache_distributed",
        BACKEND_KWARGS=backend_kwargs,
    ):
        # Using a different random_state to avoid cache hits
        model = FastICA(n_components=7, max_iter=2003)

        print("\nCalling ensemble.fit() will now print a command and wait.")
        print("Please run the printed command in a separate terminal.")
        print("The script will resume once the models are fitted by the manual worker.")

        run_ica_experiment(model, data, random_states=list(range(50)))

    print("\nManual experiment complete.")


def example_datajoint(data):
    """Demonstrates the Datajoint backend."""
    print("\n" + "=" * 50)
    print("### Example 4: Datajoint Backend ###")
    print("=" * 50)

    try:
        from dj_ml_core import login
    except ImportError:
        print(
            "DataJoint example requires ituna[datajoint] and dj_ml_core. "
            "Install with: pip install ituna[datajoint]. "
            "Note: dj_ml_core may need to be installed from the project's wheel (e.g. third_party/dj_ml_core-*.whl)."
        )
        return

    try:
        login.connect_to_database(verbose=False)
    except Exception as e:
        print("Could not connect to Datajoint database.")
        print("Please create a '.env' file based on '.env.template' and configure your database credentials.")
        print(f"Error: {e}")
        return

    print("Successfully connected to Datajoint database.")
    print("This example will create and drop a schema named 'ituna_example_schema'.")

    backend_kwargs = {
        "trigger_type": "auto",
        "num_workers": 4,
        "schema_name": "ituna_example_schema",
    }

    with ituna.config.config_context(
        DEFAULT_BACKEND="datajoint",
        BACKEND_KWARGS=backend_kwargs,
    ):
        # to make sure we start with a clean schema, we first delete it
        print("\nDropping schema 'ituna_example_schema' if it exists...")
        ituna._backends.get_backend().schema.drop(force=True)

        model = FastICA(n_components=7, max_iter=2004)

        # First run: models are fitted and cached on Datajoint.
        print("\nFirst run (fitting and caching on Datajoint):")
        run_ica_experiment(model, data)

        # Second run: models are loaded from Datajoint cache, should be much faster.
        print("\nSecond run (loading from Datajoint cache):")
        run_ica_experiment(model, data)

        # delete the schema again after the example is done
        print("\nDropping schema 'ituna_example_schema'...")
        ituna._backends.get_backend().schema.drop(force=True)
        print("Example complete.")


if __name__ == "__main__":
    # Generate a single dataset for all examples
    ica_data = generate_ica_data()

    # Run the examples
    example_in_memory(ica_data)
    example_disk_cache(ica_data)
    example_distributed_multiprocess(ica_data)
    example_datajoint(ica_data)
    example_distributed_manual(ica_data)
