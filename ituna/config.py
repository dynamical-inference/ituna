# ituna/config.py
# This file holds the global configuration for the ituna package.

import contextlib
import copy

# The default backend to use. Can be 'in_memory', 'disk_cache', 'disk_cache_distributed', 'datajoint'.
DEFAULT_BACKEND = "in_memory"
BACKEND_KWARGS = dict()
CACHE_DIR = "backend_store"
FILE_LOCK_TIMEOUT = 30  # in seconds


def _deep_update(target_dict, update_dict):
    """Recursively update nested dictionaries.

    Args:
        target_dict: The dictionary to update
        update_dict: The dictionary containing updates
    """
    for key, value in update_dict.items():
        if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
            _deep_update(target_dict[key], value)
        else:
            target_dict[key] = value


def set(**kwargs):
    """Set global configuration values by overwriting them.

    Args:
        **kwargs: Configuration key-value pairs to set.
                 All values will be strictly overwritten.
    """
    for key, value in kwargs.items():
        globals()[key.upper()] = value


def update(**kwargs):
    """Update global configuration values.

    Args:
        **kwargs: Configuration key-value pairs to update.
                 If key is 'BACKEND_KWARGS' or 'backend_kwargs', the dict will be updated recursively.
                 Otherwise, the global variable will be replaced.
    """
    for key, value in kwargs.items():
        if key.upper() == "BACKEND_KWARGS" or key.lower() == "backend_kwargs":
            # Update the BACKEND_KWARGS dict recursively instead of replacing it
            _deep_update(globals()["BACKEND_KWARGS"], value)
        else:
            # Replace the global variable
            set(**{key: value})


def get_config():
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}


@contextlib.contextmanager
def config_context(config_update_fn=set, verbose=False, **new_config):
    """Temporarily set config values."""
    old_config = copy.deepcopy(get_config())

    print(f"Storing old config: {old_config}") if verbose else None
    config_update_fn(**new_config)
    print(f"Using config: {get_config()}") if verbose else None
    try:
        yield
    finally:
        set(**old_config)
        print(f"Restored config: {get_config()}") if verbose else None


@contextlib.contextmanager
def update_config_context(verbose=False, **new_config):
    with config_context(config_update_fn=update, verbose=verbose, **new_config):
        yield


@contextlib.contextmanager
def set_config_context(verbose=False, **new_config):
    with config_context(config_update_fn=set, verbose=verbose, **new_config):
        yield
