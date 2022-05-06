from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from gym import Space
from omegaconf import DictConfig
from torch.nn import Sequential, Linear, ReLU, Tanh

from replay_buffer import ReplayBuffer


@dataclass()
class DDPGAgentConfig:
    polyak: float
    discount: float
    batch_size: int

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "DDPGAgentConfig":
        return DDPGAgentConfig(config.polyak, config.discount, config.batch_size)


class DDPGAgent:
    def __init__(
        self,
        config: DDPGAgentConfig,
        action_space: Space[np.ndarray],
        replay_buffer: ReplayBuffer,
    ) -> None:
        self.config = config
        self._action_space = action_space
        self._replay_buffer = replay_buffer
        self._policy_model = Sequential(Linear(24, 16), ReLU(), Linear(16, 4), Tanh())
        self._q_model = Sequential(
            # TODO Check how to insert policy model
            Linear(24 + 4, 16),
            ReLU(),
            Linear(16, 1),
        )
        self._target_policy_model = deepcopy(self._policy_model)
        self._target_q_model = deepcopy(self._q_model)
        self._train_mode = True

    def train_mode(self, turn_on: bool) -> None:
        self._train_mode = turn_on

    def action(self, state: np.ndarray) -> np.ndarray:
        action = self._action_space.sample()
        state_copy = np.copy(state)
        self._replay_buffer.add(state_copy, action)
        return action

    def update(self, new_state: np.ndarray, reward: float, done: bool) -> None:
        pass
