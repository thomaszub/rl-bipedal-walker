import numpy as np
from gym import Space

from replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(
        self, action_space: Space[np.ndarray], replay_buffer: ReplayBuffer
    ) -> None:
        self._action_space = action_space
        self._replay_buffer = replay_buffer

    def action(self, state: np.ndarray) -> np.ndarray:
        action = self._action_space.sample()
        state_copy = np.copy(state)
        self._replay_buffer.add(state_copy, action)
        return action

    def update(self, new_state: np.ndarray, reward: float, done: bool) -> None:
        pass
