from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


@dataclass()
class ReplayBufferConfig:
    size: int
    batch_size: int
    shuffle: bool

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "ReplayBufferConfig":
        return ReplayBufferConfig(config.size, config.batch_size, config.shuffle)


class ReplayBuffer:
    def __init__(self, config: ReplayBufferConfig) -> None:
        self.config = config
        self._replay_buffer_input = []
        self._replay_buffer_target = []

    def clear(self) -> None:
        self._replay_buffer_input = []
        self._replay_buffer_target = []

    def add(self, input: np.ndarray, target: np.ndarray) -> None:
        self._replay_buffer_input.append(input)
        self._replay_buffer_target.append(target)
        # TODO Determine insert position

    def loader(self) -> DataLoader:
        input = torch.tensor(np.array(self._replay_buffer_input)).float()
        target = torch.tensor(np.array(self._replay_buffer_target)).float()
        dataset = TensorDataset(input, target)
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle
        )
