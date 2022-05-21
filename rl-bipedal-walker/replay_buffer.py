from copy import deepcopy
from typing import Tuple

import numpy as np
import numpy.typing as npt

Batch = Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        state_shape: Tuple[int],
        action_shape: Tuple[int],
    ) -> None:
        self._size = size
        self._replay_buffer_state = np.zeros(
            (self._size,) + state_shape, dtype=np.float32
        )
        self._replay_buffer_action = np.zeros(
            (self._size,) + action_shape, dtype=np.float32
        )
        self._replay_buffer_reward = np.zeros((self._size,), dtype=np.float32)
        self._replay_buffer_done = np.zeros((self._size,), dtype=np.float32)
        self._replay_buffer_next_state = np.zeros(
            (self._size,) + state_shape, dtype=np.float32
        )
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
        self._replay_buffer_state[self._next_entry_id] = deepcopy(state)
        self._replay_buffer_action[self._next_entry_id] = deepcopy(action)
        self._replay_buffer_reward[self._next_entry_id] = deepcopy(reward)
        self._replay_buffer_done[self._next_entry_id] = deepcopy(done)
        self._replay_buffer_next_state[self._next_entry_id] = deepcopy(next_state)
        self._next_entry_id += 1
        if self._next_entry_id >= self._size:
            self._next_entry_id = 0
            self._is_full = True

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self._size, batch_size)
        return (
            self._replay_buffer_state[idx],
            self._replay_buffer_action[idx],
            self._replay_buffer_reward[idx],
            self._replay_buffer_done[idx],
            self._replay_buffer_next_state[idx],
        )

    def is_full(self) -> bool:
        return self._is_full
