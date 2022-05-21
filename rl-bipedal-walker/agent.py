import pickle
from abc import ABC, abstractmethod
from typing import List

import gym
import numpy as np
import numpy.typing as npt


class Agent(ABC):
    @abstractmethod
    def action(self, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        pass

    @abstractmethod
    def train(self, env: gym.Env) -> List[float]:
        pass

    def save(self, filename: str) -> None:
        try:
            with open(filename, "wb") as f:
                print(f"Info: Saving agent to {filename}")
                pickle.dump(self, f)
        except OSError:
            print(f"Error: Could not save agent to {filename}")

    @staticmethod
    def load(filename: str) -> "Agent":
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise ValueError(f"{filename} not found")
