from typing import Dict, Type

from glass.datasets.base import DatasetAdapter

_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register(name: str):
    def decorator(cls: Type[DatasetAdapter]):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset_class(name: str) -> Type[DatasetAdapter]:
    if name not in _REGISTRY:
        raise ValueError(f"Dataset '{name}' not found in registry. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]
