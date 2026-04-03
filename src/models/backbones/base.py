from abc import ABC, abstractmethod
from typing import Dict, Type
import numpy as np


class MoleculeEncoder(ABC):
    _registry: Dict[str, Type["MoleculeEncoder"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: Type["MoleculeEncoder"]):
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> "MoleculeEncoder":
        if name not in cls._registry:
            raise ValueError(
                f"Encoder '{name}' is not registered. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @abstractmethod
    def encode(self, input) -> np.ndarray:
        pass

    @abstractmethod
    def encode_batch(self, inputs) -> np.ndarray:
        pass
