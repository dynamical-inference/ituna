from pathlib import Path
import tempfile
import warnings

import cebra
import joblib
import numpy as np
import pytest
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

import ituna
from ituna import config
from ituna import metrics
from ituna._backends.datajoint import DatajointBackend
from ituna._test_helper import CustomConsistencyEnsemble
from ituna._test_helper import CustomConsistencyTransform


def _datajoint_packages_available():
    """Return True only if both datajoint and dj_ml_core can be imported."""
    try:
        import datajoint  # noqa: F401
        import dj_ml_core  # noqa: F401

        return True
    except ImportError:
        return False


def _database_available():
    # checks whether the database required for datajoint is available
    # assumes .env is correctly set up
    try:
        from dj_ml_core import login

        login.connect_to_database(verbose=True)
        return True
    except Exception as e:
        print(f"Database not available: {e}")
        return False


def _datajoint_available():
    """Return True only if datajoint packages are installed and database is available."""
    if not _datajoint_packages_available():
        return False
    return _database_available()


@pytest.mark.skipif(not _datajoint_available(), reason="datajoint/dj_ml_core not installed or database not available")
def test_datajoint_backend_picklable():
    """Check that DatajointBackend is picklable with joblib."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_cache_dir = tmpdir / "cache"

        backend = DatajointBackend(schema_name="test_schema", cache_dir=tmp_cache_dir)
        filename = tmpdir / "backend.pkl"
        joblib.dump(backend, filename)
        loaded_backend = joblib.load(filename)
        assert loaded_backend is not None
        assert loaded_backend.schema_name == backend.schema_name
        assert loaded_backend.schema is not None
        assert loaded_backend.tables is not None


def get_estimator(estimator_cls):
    if estimator_cls == FastICA:
        estimator = FastICA(n_components=7, random_state=42, max_iter=200)
    elif estimator_cls == PCA:  # PCA
        estimator = PCA(n_components=7, random_state=42)

    elif estimator_cls == cebra.CEBRA:  # CEBRA
        estimator = cebra.CEBRA(
            model_architecture="offset1-model",
            batch_size=1,
            learning_rate=3e-4,
            temperature=1,
            output_dimension=7,
            max_iterations=1,
            distance="cosine",
            conditional="time_delta",
            device="cuda_if_available",
            verbose=True,
            time_offsets=1,
        )
    else:
        raise ValueError(f"Unsupported base model class: {estimator_cls}")

    return estimator


@pytest.mark.skipif(not _datajoint_available(), reason="datajoint/dj_ml_core not installed or database not available")
@pytest.mark.parametrize(
    "estimator_cls",
    [
        FastICA,
        PCA,
        cebra.CEBRA,
    ],
)
def test_multi_process(ica_data, estimator_cls):
    """Test that the DiskCacheDistributedBackend caches fitted models."""

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        backend_kwargs = {
            "trigger_type": "auto",
            "num_workers": 3,
            "fit_time_out": 60,
            "schema_name": "pytest_ituna",
        }

        with config.config_context(
            DEFAULT_BACKEND="datajoint",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS=backend_kwargs,
        ):
            # to make sure we start with a clean schema, we first delete it
            ituna._backends.get_backend().schema.drop(force=True)
            estimator = get_estimator(estimator_cls)

            ensemble = ituna.ConsistencyEnsemble(
                estimator=estimator,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=3,
            )

            # First fit, should train and cache models
            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble.fit(ica_data)
                score1 = ensemble.score(ica_data)
                transform1 = ensemble.transform(ica_data)

            # Check that cache directory is not empty
            model_cache_dir = ensemble._backend._trained_models_dir
            assert model_cache_dir.exists()
            cached_files = list(model_cache_dir.glob("*.pkl"))
            # +1 for the consistency transform which is now also cached
            assert len(cached_files) == ensemble.random_states + 1

            estimator2 = get_estimator(estimator_cls)

            ensemble2 = ituna.ConsistencyEnsemble(
                estimator=estimator2,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=3,
            )

            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble2.fit(ica_data)
                score2 = ensemble2.score(ica_data)
                transform2 = ensemble2.transform(ica_data)

            np.testing.assert_allclose(score1, score2)
            np.testing.assert_allclose(transform1, transform2)
            assert isinstance(score1, (int, float, np.number))

            # delete the schema again after the test is done
            ensemble._backend.schema.drop(force=True)


@pytest.mark.skipif(not _datajoint_available(), reason="datajoint/dj_ml_core not installed or database not available")
@pytest.mark.parametrize(
    "estimator_cls",
    [
        FastICA,
        PCA,
    ],
)
def test_model_as_data_argument(ica_data, estimator_cls):
    """Test that the DiskCacheDistributedBackend caches fitted models."""

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        backend_kwargs = {
            "trigger_type": "auto",
            "num_workers": 3,
            "fit_time_out": 60,
            "schema_name": "pytest_ituna",
        }

        with config.config_context(
            DEFAULT_BACKEND="datajoint",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS=backend_kwargs,
        ):
            # to make sure we start with a clean schema, we first delete it
            ituna._backends.get_backend().schema.drop(force=True)
            estimator = get_estimator(estimator_cls)

            ensemble = CustomConsistencyEnsemble(
                estimator=estimator,
                consistency_transform=CustomConsistencyTransform(),
                random_states=3,
            )

            # First fit, should train and cache models
            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble.fit(ica_data)
                score1 = ensemble.score(ica_data)
                transform1 = ensemble.transform(ica_data)

            # Check that cache directory is not empty
            model_cache_dir = ensemble._backend._trained_models_dir
            assert model_cache_dir.exists()
            cached_files = list(model_cache_dir.glob("*.pkl"))
            # +1 for the consistency transform which is now also cached
            assert len(cached_files) == ensemble.random_states + 1

            estimator2 = get_estimator(estimator_cls)

            ensemble2 = CustomConsistencyEnsemble(
                estimator=estimator2,
                consistency_transform=CustomConsistencyTransform(),
                random_states=3,
            )

            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble2.fit(ica_data)
                score2 = ensemble2.score(ica_data)
                transform2 = ensemble2.transform(ica_data)

            np.testing.assert_allclose(score1, score2)
            np.testing.assert_allclose(transform1, transform2)
            assert isinstance(score1, (int, float, np.number))

            # delete the schema again after the test is done
            ensemble._backend.schema.drop(force=True)
