from dataclasses import dataclass
from typing import List

import gym
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from tqdm import trange

from agent import Agent


@dataclass()
class ESAgentConfig:
    generations: int

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "ESAgentConfig":
        return ESAgentConfig(config.generations)


class ESAgent(Agent):
    def __init__(self, config: ESAgentConfig) -> None:
        self.config = config

    def action(self, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.random.uniform(-1.0, 1.0, size=4)

    def train(self, env: gym.Env) -> List[float]:
        rewards = []
        sum_reward = 0
        with trange(0, self.config.generations) as tgen:
            for generation in tgen:
                tgen.set_postfix(generation=generation, last_sum_reward=sum_reward)
                state, done, sum_reward = env.reset(), False, 0
                while not done:
                    action = self.action(state)

                    new_state, reward, done, _ = env.step(action)
                    sum_reward += reward

                    # TODO Optimize
                    state = new_state

                rewards.append(sum_reward)

        return rewards
