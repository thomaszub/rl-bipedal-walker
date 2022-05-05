from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import TensorDataset


@dataclass()
class ReplayBufferConfig:
    size: int

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "ReplayBufferConfig":
        return ReplayBufferConfig(config.size)


class ReplayBuffer:
    def __init__(
        self,
        config: ReplayBufferConfig,
        input_shape: Tuple[int],
        target_shape: Tuple[int],
    ) -> None:
        self.config = config
        self._replay_buffer_input = np.zeros((self.config.size,) + input_shape)
        self._replay_buffer_target = np.zeros((self.config.size,) + target_shape)
        self._next_entry_id = 0
        self._is_full = False

    def add(self, input: np.ndarray, target: np.ndarray) -> None:
        self._replay_buffer_input[self._next_entry_id] = input
        self._replay_buffer_target[self._next_entry_id] = target
        self._next_entry_id += 1
        if self._next_entry_id >= self.config.size:
            self._next_entry_id = 0
            self._is_full = True

    def is_full(self) -> bool:
        return self._is_full

    def dataset(self) -> TensorDataset:
        input = torch.tensor(self._replay_buffer_input).float()
        target = torch.tensor(self._replay_buffer_target).float()
        return TensorDataset(input, target)
