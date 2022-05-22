from typing import Callable, List

import numpy as np
import numpy.typing as npt
import torch

Layer = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]

Activation = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]


def polyak_update(
    source: torch.nn.Module, target: torch.nn.Module, polyak: float
) -> None:
    for p_tgt, p_src in zip(target.parameters(), source.parameters()):
        p_tgt.copy_(polyak * p_tgt + (1.0 - polyak) * p_src)


def relu(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.maximum(x, 0)


class LinearLayer(Layer):
    def __init__(
        self, in_size: int, out_size: int, activation: Activation = None
    ) -> None:
        self.size = (in_size, out_size)
        self.W = np.random.uniform(-1.0, 1.0, size=(in_size, out_size)).astype(
            np.float32
        )
        self.b = np.random.uniform(-1.0, 1.0, size=out_size).astype(np.float32)
        self.activation = activation
        if activation is not None:
            self._func = lambda x: self.activation(np.add(np.matmul(x, self.W), self.b))
        else:
            self._func = lambda x: np.add(np.matmul(x, self.W), self.b)

    def __call__(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self._func(input)


class SequentialLayer(Layer):
    def __init__(self, *layers: List[Layer]) -> None:
        self._layers = layers

    def __call__(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        for layer in self._layers:
            input = layer(input)
        return input
