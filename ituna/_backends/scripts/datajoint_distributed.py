import argparse
import json
import os
from typing import List, Optional

from dotenv import find_dotenv
from dotenv import load_dotenv

from ituna._backends.datajoint import DatajointBackend

load_dotenv(
    find_dotenv(usecwd=True),
    override=True,
    verbose=True,
)


def run_fit(
    sweep_name: List[str],
    schema_name: Optional[str],
    host: Optional[str],
    user: Optional[str],
    password: Optional[str],
    cache_dir: Optional[str],
    partial_match: bool,
    order: str,
    limit: Optional[int],
    suppress_errors: bool,
    make_kwargs: Optional[str],
):
    """Initialize the backend and fit models from a sweep.

    This function connects to the DataJoint database, initializes the
    `DatajointBackend`, and calls the `fit_sweep_models` method to train
    the models for the specified sweep.

    Parameters
    ----------
    sweep_name : list of str
        The UUID(s) of the sweep(s) to process.
    schema_name : str, optional
        The name of the database schema.
    host : str, optional
        The database host.
    user : str, optional
        The database user.
    password : str, optional
        The database password.
    datajoint_cache_dir : str, optional
        The directory for caching DataJoint models and data.
    partial_match : bool
        If True, `sweep_name` is treated as a pattern.
    order : {'original', 'reverse', 'random'}
        The order in which to process training jobs.
    limit : int, optional
        The maximum number of models to train.
    suppress_errors : bool
        If True, suppress errors during training.
    make_kwargs : str, optional
        A JSON string of keyword arguments for the `make` method.
    """
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"Connecting to schema: {schema_name}", flush=True)

    backend_kwargs = {
        "schema_name": schema_name,
        "host": host,
        "user": user,
        "password": password,
    }
    if cache_dir is not None:
        backend_kwargs["cache_dir"] = cache_dir

    backend = DatajointBackend(**backend_kwargs)

    parsed_make_kwargs = json.loads(make_kwargs) if make_kwargs else None

    print(f"Fitting models for sweep(s): {sweep_name}", flush=True)
    backend.fit_sweep_models(
        sweep_name=sweep_name,
        limit=limit,
        order=order,
        suppress_errors=suppress_errors,
        partial_match=partial_match,
        make_kwargs=parsed_make_kwargs,
    )
    print("Fitting complete.", flush=True)


def main():
    """
    CLI entry point for fitting models.
    """
    parser = argparse.ArgumentParser(description="Fit models from a sweep using DatajointBackend.")
    parser.add_argument(
        "--sweep-name",
        type=str,
        nargs="+",
        required=True,
        help="One or more sweep UUIDs to process.",
    )
    parser.add_argument(
        "--schema-name",
        type=str,
        default=None,
        help="Name of the datajoint schema.",
    )
    parser.add_argument("--host", type=str, default=None, help="Datajoint host.")
    parser.add_argument("--user", type=str, default=None, help="Datajoint user.")
    parser.add_argument("--password", type=str, default=None, help="Datajoint password.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Path to the datajoint cache directory.",
    )
    parser.add_argument(
        "--partial-match",
        action="store_true",
        help="Allow partial matching for sweep UUIDs.",
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["original", "reverse", "random"],
        default="random",
        help="Order in which to process models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of models to process.",
    )
    parser.add_argument(
        "--suppress-errors",
        action="store_true",
        help="Suppress errors during model fitting.",
    )
    parser.add_argument(
        "--make-kwargs",
        type=str,
        default=None,
        help="JSON string of keyword arguments for the make function.",
    )

    args = parser.parse_args()
    run_fit(
        sweep_name=args.sweep_name,
        schema_name=args.schema_name,
        host=args.host,
        user=args.user,
        password=args.password,
        cache_dir=args.cache_dir,
        partial_match=args.partial_match,
        order=args.order,
        limit=args.limit,
        suppress_errors=args.suppress_errors,
        make_kwargs=args.make_kwargs,
    )


if __name__ == "__main__":
    main()
