import numpy as np
from gym import Space


class DDPGAgent:
    def __init__(self, action_space: Space[np.ndarray]) -> None:
        self._action_space = action_space

    def action(self) -> np.ndarray:
        return self._action_space.sample()
