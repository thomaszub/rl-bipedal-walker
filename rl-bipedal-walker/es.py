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
    steps: int

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "ESAgentConfig":
        return ESAgentConfig(config.steps)


class ESAgent(Agent):
    def __init__(self, config: ESAgentConfig) -> None:
        self.config = config

    def action(self, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.random.uniform(-1.0, 1.0, size=4)

    def train(self, env: gym.Env) -> List[float]:
        rewards = []
        state, done, sum_reward = env.reset(), False, 0
        curr_episode = 1
        with trange(0, self.config.steps) as tr:
            tr.set_postfix(curr_episode=curr_episode, last_sum_reward=sum_reward)
            for _ in tr:
                action = self.action(state)

                new_state, reward, done, _ = env.step(action)
                sum_reward += reward

                # TODO Optimize
                if done:
                    curr_episode += 1
                    rewards.append(sum_reward)
                    tr.set_postfix(
                        curr_episode=curr_episode, last_sum_reward=sum_reward
                    )
                    state, done, sum_reward = env.reset(), False, 0
                else:
                    state = new_state

        return rewards
