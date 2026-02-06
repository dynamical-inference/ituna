from ituna._backends import utils
from ituna._backends.base import Backend
from ituna._backends.datajoint import DatajointBackend
from ituna._backends.disk_cache import DiskCacheBackend
from ituna._backends.disk_cache import DiskCacheDistributedBackend
from ituna._backends.in_memory import InMemoryBackend

# backend registry
_BACKENDS = {
    "in_memory": InMemoryBackend,
    "disk_cache": DiskCacheBackend,
    "disk_cache_distributed": DiskCacheDistributedBackend,
    "datajoint": DatajointBackend,
}


# backend factory
def get_backend(backend_name: str = None):
    """
    Factory function to get a backend instance.

    If backend_name is None, it uses the default from the global config.
    """
    # delayed import so it uses the updated config
    from ituna import config

    if backend_name is None:
        backend_name = config.DEFAULT_BACKEND

    if backend_name not in _BACKENDS:
        raise ValueError(f"Unknown backend: '{backend_name}'. Available backends are: {list(_BACKENDS.keys())}")

    backend_factory = _BACKENDS[backend_name]

    kwargs = {}
    if backend_name == "disk_cache" and config.CACHE_DIR:
        kwargs["cache_dir"] = config.CACHE_DIR
    elif backend_name == "disk_cache_distributed" and config.CACHE_DIR:
        kwargs["cache_dir"] = config.CACHE_DIR
        if config.BACKEND_KWARGS:
            kwargs.update(config.BACKEND_KWARGS)
    elif backend_name == "datajoint" and config.BACKEND_KWARGS:
        kwargs["cache_dir"] = config.CACHE_DIR
        if config.BACKEND_KWARGS:
            kwargs.update(config.BACKEND_KWARGS)
    return backend_factory(**kwargs)


__all__ = [
    "get_backend",
    "Backend",
    "utils",
    "DiskCacheBackend",
    "DiskCacheDistributedBackend",
    "InMemoryBackend",
    "DatajointBackend",
]
