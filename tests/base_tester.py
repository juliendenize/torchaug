from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseTesterFunctional(ABC):
    def get_generator(self, seed: int = 28) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed)

        return g

    def get_uint8_image(
        self, shape: list[int], generator: torch.Generator | None = None
    ) -> Tensor:
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)

    def get_float_image(
        self,
        shape: list[int],
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return torch.rand(shape, dtype=dtype, generator=generator)

    @abstractmethod
    def test_output_values_uint8(self):
        pass

    @abstractmethod
    def test_output_values_float(self):
        pass

    @abstractmethod
    def test_wrong_input_tensor(self):
        pass

    @abstractmethod
    def test_compile(self):
        pass


class BaseTesterTransform(ABC):
    def get_generator(self, seed: int = 28) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed)

        return g

    def get_uint8_image(
        self, shape: list[int], generator: torch.Generator | None = None
    ) -> Tensor:
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)

    def get_float_image(
        self,
        shape: list[int],
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return torch.rand(shape, dtype=dtype, generator=generator)

    @abstractmethod
    def test_functional_float(self):
        pass

    @abstractmethod
    def test_functional_uint8(self):
        pass

    @abstractmethod
    def test_repr(self):
        pass

    @abstractmethod
    def test_compile(self):
        pass
