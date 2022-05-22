from dataclasses import dataclass
from typing import List

import gym
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from tqdm import trange

from agent import Agent
from core import Layer, LinearLayer, SequentialLayer, relu


@dataclass()
class EvolutionalAgentConfig:
    generations: int

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "EvolutionalAgentConfig":
        return EvolutionalAgentConfig(config.generations)


class EvolutionalAgent(Agent):
    def __init__(self, config: EvolutionalAgentConfig) -> None:
        self.config = config
        self._policy_model = SequentialLayer(
            LinearLayer(24, 64, activation=relu),
            LinearLayer(64, 32, activation=relu),
            LinearLayer(32, 4, activation=np.tanh),
        )

    def action(self, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self._policy_model(state)

    def train(self, env: gym.Env) -> List[float]:
        rewards = []
        sum_reward = 0
        with trange(0, self.config.generations) as tgen:
            for generation in tgen:
                tgen.set_postfix(generation=generation, last_sum_reward=sum_reward)

                sum_reward = self._play(env, self._policy_model)

                rewards.append(sum_reward)

        return rewards

    def _play(self, env: gym.Env, policy_model: Layer) -> float:
        state, done, sum_reward = env.reset(), False, 0
        while not done:
            action = policy_model(state)
            new_state, reward, done, _ = env.step(action)
            sum_reward += reward
            state = new_state

        return sum_reward
