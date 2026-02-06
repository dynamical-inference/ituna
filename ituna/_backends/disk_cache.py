import csv
import pathlib
import traceback
from typing import Any, List, Literal, Optional, Tuple, Union

import pandas as pd
import sklearn
import sklearn.base
import typeguard

from ituna import estimator
from ituna import metrics
from ituna._backends import base
from ituna._backends import in_memory
from ituna._backends import utils


@typeguard.typechecked
class DiskCacheBackend(base.Backend):
    def __init__(self, cache_dir: Union[str, pathlib.Path], **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.trained_models_cache = self.cache_dir / "trained_models"
        self.trained_models_cache.mkdir(parents=True, exist_ok=True)

    def _hash_model(self, model, model_id: Optional[int] = None):
        # Only pass model_id if model is non-deterministic
        kwargs = {}
        if estimator.check_non_deterministic(model):
            assert model_id is not None, "Model ID is required for non-deterministic models"
            kwargs["model_id"] = model_id

        return utils.hash_sklearn(model, **kwargs)

    def _hash_model_data(self, model, data, model_id: Optional[int] = None):
        # only pass model_id if model is non-deterministic
        model_hash = self._hash_model(model, model_id=model_id)

        return utils.hash_model_data(model_hash=model_hash, data=data)

    @typeguard.typechecked
    def _retrieve_trained_model(self, model_data_hash: str) -> Optional[sklearn.base.BaseEstimator]:
        """
        Retrieve a model from the cache if it exists, otherwise return None.
        """
        # Fall back to joblib serialization
        model_path = self.trained_models_cache / f"{model_data_hash}"

        try:
            model = utils.load_model(model_path)
        except FileNotFoundError:
            model = None

        return model

    @typeguard.typechecked
    def _store_trained_model(
        self,
        model: sklearn.base.BaseEstimator,
        data,
        model_data_hash: Optional[str] = None,
        model_id: Optional[int] = None,
    ) -> Tuple[str, pathlib.Path]:
        """
        Store a model in the cache.
        """

        if model_data_hash is None:
            model_data_hash = self._hash_model_data(
                model,
                data,
                model_id=model_id,
            )

        model_path = self.trained_models_cache / f"{model_data_hash}"
        utils.store_model(model_path, model)
        return model_data_hash, model_path

    @typeguard.typechecked
    def fit_models(
        self,
        models: List[sklearn.base.BaseEstimator],
        *args,
        **kwargs,
    ) -> List[sklearn.base.BaseEstimator]:
        """
        Fit models in the queue.
        """
        if any(isinstance(model, metrics.ConsistencyTransform) for model in models):
            # fall back to in memory backend for anything that is not a ConsistencyEnsemble
            return in_memory.InMemoryBackend().fit_models(models, *args, **kwargs)

        data = utils.DataArguments(*args, **kwargs)
        trained_models = []
        for i, model in enumerate(models):
            # combine data and model hash to identify models trained on this data
            model_data_hash = self._hash_model_data(model, data, model_id=i)
            # try retrieving model from cache
            trained_model = self._retrieve_trained_model(model_data_hash)
            if trained_model is None:
                trained_model = model.fit(*data.args, **data.kwargs)
                self._store_trained_model(trained_model, data, model_data_hash=model_data_hash)

            trained_models.append(trained_model)

        return trained_models


@typeguard.typechecked
class DiskCacheDistributedBackend(DiskCacheBackend, base.DistributedComputationMixin):
    """
    Backend for caching models on disk and computing model fit distributed in separate processes.
    """

    def __init__(
        self,
        cache_dir: Union[str, pathlib.Path],
        trigger_type: Literal["auto", "manual"] = "auto",
        order_by: Literal[
            "data_hash",
            "model_hash",
            "random",
            "sweep_name",
        ] = "random",
        num_workers: Optional[int] = 1,
        progress_bar: bool = True,
        sweep_type: Literal["uuid", "constant"] = "uuid",
        sweep_name: Optional[str] = None,
        cache_refresh_interval: float = 1.0,
        fit_time_out: Optional[float] = None,
    ):
        super().__init__(
            cache_dir=cache_dir,
            trigger_type=trigger_type,
            num_workers=num_workers,
            progress_bar=progress_bar,
            sweep_type=sweep_type,
            sweep_name=sweep_name,
            cache_refresh_interval=cache_refresh_interval,
            fit_time_out=fit_time_out,
        )
        # add data cache
        self.data_cache = self.cache_dir / "data"
        self.data_cache.mkdir(parents=True, exist_ok=True)
        # add trained models cache
        self.model_cache = self.cache_dir / "models"
        self.model_cache.mkdir(parents=True, exist_ok=True)

        self.sweep_cache = self.cache_dir / "sweep_data"
        self.sweep_cache.mkdir(parents=True, exist_ok=True)

        self._worker_log_dir = self.cache_dir / "worker_logs"
        self._worker_log_dir.mkdir(parents=True, exist_ok=True)

        self.order_by = order_by

    def _add_to_sweep(
        self,
        sweep_name: str,
        model_hash: str,
        data_hash: str,
        model_data_hash: str,
    ):
        """
        Add a model to the sweep.
        """

        # sanity check recompute model_data_hash
        assert model_data_hash == utils.hash_model_data(model_hash=model_hash, data_hash=data_hash), "model_data_hash does not match"
        # Create sweep file path
        sweep_file = self.sweep_cache / f"{sweep_name}.csv"

        with utils.file_lock_context(sweep_file):
            # Read current sweep data to check if row already exists
            try:
                current_df = self._read_sweep_csv_single(sweep_name, no_lock=True)
                # Check if this exact row already exists
                row_exists = (current_df["model_data_hash"] == model_data_hash).any()
            except FileNotFoundError:
                # File doesn't exist yet, so row doesn't exist
                row_exists = False

            # Only add the row if it doesn't already exist
            if not row_exists:
                # Write header if file doesn't exist, then append the row
                file_exists = sweep_file.exists()
                with open(sweep_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(["data_hash", "model_hash", "model_data_hash"])
                    writer.writerow([data_hash, model_hash, model_data_hash])

    def _read_sweep_csv_single(self, sweep_name: str, no_lock: bool = False) -> pd.DataFrame:
        """
        Read a single sweep CSV file and return as a pandas DataFrame.

        Args:
            sweep_name: The UUID of the sweep to read

        Returns:
            pandas.DataFrame: DataFrame containing the sweep data with columns
                             ['data_hash', 'model_hash', 'model_data_hash', 'sweep_name']
        """
        sweep_file = self.sweep_cache / f"{sweep_name}.csv"
        if not sweep_file.exists():
            raise FileNotFoundError(f"Sweep file {sweep_file} does not exist")
        if not no_lock:
            with utils.file_lock_context(sweep_file):
                df = pd.read_csv(sweep_file)
        else:
            df = pd.read_csv(sweep_file)

        df["sweep_name"] = sweep_name
        return df

    def _read_sweep_csvs(self, sweep_names: Union[str, List[str]]) -> pd.DataFrame:
        """
        Read one or more sweep CSV files and return as a combined pandas DataFrame.

        Args:
            sweep_names: Single sweep UUID string or list of sweep UUIDs to read

        Returns:
            pandas.DataFrame: DataFrame containing the combined sweep data with columns
                             ['data_hash', 'model_hash', 'model_data_hash', 'sweep_name']
        """
        if isinstance(sweep_names, str):
            sweep_names = [sweep_names]

        dfs = []
        for sweep_name in sweep_names:
            df = self._read_sweep_csv_single(sweep_name)
            dfs.append(df)

        if not dfs:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["data_hash", "model_hash", "model_data_hash", "sweep_name"])

        return pd.concat(dfs, ignore_index=True)

    def _get_fit_distributed_cmd(self, sweep_name: str):
        """
        Get the command to run the fit_distributed script.
        """
        return [
            "ituna-fit-distributed",
            "--sweep-name",
            sweep_name,
            "--cache-dir",
            str(self.cache_dir.resolve()),
            "--order-by",
            self.order_by,
        ]

    @typeguard.typechecked
    def _retrieve_data(self, data_hash: str) -> Optional[Any]:
        """
        Retrieve data from the cache if it exists, otherwise return None.
        """
        data_path = self.data_cache / f"{data_hash}"
        return utils.load_data(data_path)

    @typeguard.typechecked
    def _store_data(self, data) -> Tuple[str, pathlib.Path]:
        """
        Store data in the cache. Data must be a numpy array.
        """

        data_hash = utils.hash_data(data)
        data_path = self.data_cache / f"{data_hash}"
        utils.store_data(data_path, data)
        return data_hash, data_path

    @typeguard.typechecked
    def _retrieve_model(self, model_hash: str) -> Optional[sklearn.base.BaseEstimator]:
        """
        Retrieve a model from the model cache if it exists, otherwise return None.
        """
        model_path = self.model_cache / f"{model_hash}"
        return utils.load_model(model_path)

    @typeguard.typechecked
    def _store_model(
        self,
        model: sklearn.base.BaseEstimator,
        model_id: Optional[int] = None,
    ) -> Tuple[str, pathlib.Path]:
        """
        Store a model in the model cache.
        """
        model_hash = self._hash_model(model, model_id=model_id)
        model_path = self.model_cache / f"{model_hash}"
        utils.store_model(model_path, model)
        return model_hash, model_path

    # @typeguard.typechecked
    def fit_models(
        self,
        models: List[sklearn.base.BaseEstimator],
        *args,
        **kwargs,
    ) -> List[Union[sklearn.base.BaseEstimator, None]]:
        """
        Fit models in the queue.
        """

        if any(isinstance(model, metrics.ConsistencyTransform) for model in models):
            # fall back to in memory backend for anything that is not a ConsistencyEnsemble
            return in_memory.InMemoryBackend().fit_models(models, *args, **kwargs)

        data = utils.DataArguments(*args, **kwargs)

        # add data to cache
        data_hash, _ = self._store_data(data)

        training_queue = {self._hash_model_data(model, data, model_id=i): model for i, model in enumerate(models)}

        sweep_name = self._get_sweep_name()

        for i, (model_data_hash, model) in enumerate(training_queue.items()):
            model_hash, _ = self._store_model(model, model_id=i)
            self._add_to_sweep(
                model_hash=model_hash,
                data_hash=data_hash,
                model_data_hash=model_data_hash,
                sweep_name=sweep_name,
            )

        self._trigger_sweep(sweep_name)

        # wait for sweep to finish
        trained_models = self._collect_trained_models(sweep_name, list(training_queue.keys()))

        return trained_models

    def _get_sweep_status(
        self,
        sweep_names: Union[str, List[str]],
        models_of_interest: Optional[List[str]] = None,
    ):
        """
        Get the status of a sweep.
        """
        if isinstance(sweep_names, str):
            sweep_names = [sweep_names]

        sweep_data = self._read_sweep_csvs(sweep_names)
        status = {}

        # Calculate status for the entire sweep
        all_hashes = sweep_data["model_data_hash"].tolist()
        sweep_trained, sweep_errors, sweep_reserved = 0, 0, 0
        for h in all_hashes:
            # Check for both regular joblib files and custom serialization directories
            pkl_file = (self.trained_models_cache / h).with_suffix(".pkl")
            if pkl_file.exists():
                sweep_trained += 1
            elif (self.trained_models_cache / f"error_{h}.log").exists():
                sweep_errors += 1
            elif (self.trained_models_cache / f"reserved_{h}").exists():
                sweep_reserved += 1

        status["sweep_total"] = len(all_hashes)
        status["sweep_trained"] = sweep_trained
        status["sweep_errors"] = sweep_errors
        status["sweep_reserved"] = sweep_reserved
        status["sweep_completed"] = sweep_trained + sweep_errors

        # Calculate status for models of interest if provided
        if models_of_interest:
            interest_hashes = set(models_of_interest)
            trained, errors, reserved = 0, 0, 0
            for h in interest_hashes:
                # Check for both regular joblib files and custom serialization directories
                pkl_file = (self.trained_models_cache / h).with_suffix(".pkl")
                if pkl_file.exists():
                    trained += 1
                elif (self.trained_models_cache / f"error_{h}.log").exists():
                    errors += 1
                elif (self.trained_models_cache / f"reserved_{h}").exists():
                    reserved += 1

            status["total"] = len(interest_hashes)
            status["trained"] = trained
            status["errors"] = errors
            status["reserved"] = reserved
            status["completed"] = trained + errors

        return status

    @typeguard.typechecked
    def fit_sweep_models(
        self,
        sweep_name: Union[str, List[str]],
        limit: Optional[int] = None,
        raise_error: bool = False,
    ):
        """Fit models in the sweep using distributed processing.

        This method processes models in a sweep by loading them from cache,
        fitting them with data, and storing the trained models. It includes
        reservation mechanisms to prevent duplicate processing in distributed
        environments.

        Parameters
        ----------
        sweep_name : str or list of str
            UUID(s) identifying the sweep(s) to process.
        limit : int, optional
            Maximum number of models to process. If None, all models
            in the sweep are processed.

        Notes
        -----
        The method implements a reservation system to handle concurrent
        processing:

        1. Fetches sweep data from CSV files
        2. Iterates over each model-data pair
        3. Checks if model is already trained or reserved
        4. Creates reservation file to prevent duplicate processing
        5. Loads model and data from cache
        6. Fits the model and stores the trained result
        7. Handles errors by logging them to error files

        Reservation files are named `reserved_{model_data_hash}` and error
        files are named `error_{model_data_hash}.log` in the trained_models
        cache directory.

        Raises
        ------
        ValueError
            If model or data cannot be found in cache when expected.
        """

        # Read sweep data using _read_sweep_csvs
        sweep_data = self._read_sweep_csvs(sweep_name)

        # Sort the sweep data according to the order_by parameter
        if self.order_by == "random":
            sweep_data = sweep_data.sample(frac=1).reset_index(drop=True)
        elif self.order_by in ["data_hash", "model_hash", "sweep_name"]:
            sweep_data = sweep_data.sort_values(by=self.order_by).reset_index(drop=True)

        # Track number of models fitted
        models_fitted = 0

        for _, row in sweep_data.iterrows():
            model_hash = row["model_hash"]
            data_hash = row["data_hash"]
            model_data_hash = row["model_data_hash"]

            # Check if model is already trained
            trained_file = (self.trained_models_cache / model_data_hash).with_suffix(".pkl")
            if trained_file.exists():
                continue

            # Check if model is reserved or has error
            reserved_file = self.trained_models_cache / f"reserved_{model_data_hash}"
            error_file = self.trained_models_cache / f"error_{model_data_hash}.log"

            if reserved_file.exists() or error_file.exists():
                continue

            # Check if we've reached the limit
            if limit is not None and models_fitted >= limit:
                break

            reserved_file.touch()

            try:
                # Load model from cache
                model = self._retrieve_model(model_hash)
                if model is None:
                    raise ValueError(f"Model with hash {model_hash} not found in cache")

                # Load data from cache
                fit_data = self._retrieve_data(data_hash)
                if fit_data is None:
                    raise ValueError(f"Data with hash {data_hash} not found in cache")

                # Fit the model
                trained_model = model.fit(*fit_data.args, **fit_data.kwargs)

                # Store trained model
                self._store_trained_model(trained_model, fit_data, model_data_hash=model_data_hash)

                # Increment counter after successful fit
                models_fitted += 1

            except Exception as e:
                # Write error to log file
                with utils.file_lock_context(error_file):
                    with open(error_file, "w") as f:
                        f.write(f"Error: {str(e)}\n")
                        f.write(f"Traceback:\n{traceback.format_exc()}")
                if raise_error:
                    raise e
            finally:
                # Remove reservation file
                with utils.file_lock_context(reserved_file):
                    if reserved_file.exists():
                        reserved_file.unlink()
