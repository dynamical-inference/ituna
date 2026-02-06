import argparse
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv
from dotenv import load_dotenv

from ituna._backends.disk_cache import DiskCacheDistributedBackend

load_dotenv(
    find_dotenv(usecwd=True),
    override=False,
    verbose=True,
)


def run_fit(
    sweep_name: List[str],
    cache_dir: Optional[Path] = None,
    order_by: str = "random",
    limit: Optional[int] = None,
):
    """
    Initialize the backend and fit models from a sweep.
    """
    backend_kwargs = {
        "trigger_type": "auto",
        "order_by": order_by,
    }
    if cache_dir is not None:
        backend_kwargs["cache_dir"] = cache_dir

    backend = DiskCacheDistributedBackend(**backend_kwargs)
    backend.fit_sweep_models(sweep_name=sweep_name, limit=limit)


def main():
    """
    CLI entry point for fitting models.
    """
    parser = argparse.ArgumentParser(description="Fit models from a sweep using DiskCacheDistributedBackend.")
    parser.add_argument(
        "--sweep-name",
        type=str,
        nargs="+",
        required=True,
        help="One or more sweep UUIDs to process.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Path to the cache directory (optional).",
    )
    parser.add_argument(
        "--order-by",
        type=str,
        choices=["data_hash", "model_hash", "random", "sweep_name"],
        default="random",
        help="Order in which to process models (optional, default: random).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of models to process (optional).",
    )

    args = parser.parse_args()
    run_fit(
        sweep_name=args.sweep_name,
        cache_dir=args.cache_dir,
        order_by=args.order_by,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
