from dataclasses import dataclass
from typing import Tuple

import numpy as np
from omegaconf import DictConfig


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
        state_shape: Tuple[int],
        action_shape: Tuple[int],
    ) -> None:
        self.config = config
        self._replay_buffer_state = np.zeros((self.config.size,) + state_shape)
        self._replay_buffer_action = np.zeros((self.config.size,) + action_shape)
        self._replay_buffer_reward = np.zeros((self.config.size,))
        self._replay_buffer_done = np.zeros((self.config.size,))
        self._replay_buffer_next_state = np.zeros((self.config.size,) + state_shape)
        self._next_entry_id = 0
        self._is_full = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_state: np.ndarray,
    ) -> None:
        self._replay_buffer_state[self._next_entry_id] = state
        self._replay_buffer_action[self._next_entry_id] = action
        self._replay_buffer_reward[self._next_entry_id] = reward
        self._replay_buffer_done[self._next_entry_id] = done
        self._replay_buffer_next_state[self._next_entry_id] = next_state
        self._next_entry_id += 1
        if self._next_entry_id >= self.config.size:
            self._next_entry_id = 0
            self._is_full = True

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, float, bool, np.ndarray]:
        def choice(buffer: np.ndarray) -> np.ndarray:

            # idx = np.random.choice(self.config.size, (batch_size))
            # return np.take(buffer, idx)
            rng = np.random.default_rng()
            return rng.choice(buffer, batch_size, replace=False)

        # TODO Fehler
        return (
            choice(self._replay_buffer_state),
            choice(self._replay_buffer_action),
            choice(self._replay_buffer_reward),
            choice(self._replay_buffer_done),
            choice(self._replay_buffer_next_state),
        )

    def is_full(self) -> bool:
        return self._is_full
