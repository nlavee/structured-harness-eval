from typing import Dict, Type

from glass.metrics.base import BaseMetric

_REGISTRY: Dict[str, Type[BaseMetric]] = {}


def register(name: str):
    def decorator(cls: Type[BaseMetric]):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_metric_class(name: str) -> Type[BaseMetric]:
    if name not in _REGISTRY:
        raise ValueError(f"Metric '{name}' not found in registry. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]
