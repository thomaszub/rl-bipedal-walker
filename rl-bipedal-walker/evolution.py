from copy import deepcopy
from dataclasses import dataclass
from typing import List

import gym
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from tqdm import trange

from agent import Agent
from core import Layer, LinearLayer, Model, relu


@dataclass(frozen=True)
class Candidate:
    model: Layer
    fitness: float


@dataclass()
class EvolutionalAgentConfig:
    generations: int
    mutations_per_parent: int
    mutation_strength: float

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "EvolutionalAgentConfig":
        return EvolutionalAgentConfig(
            config.generations, config.mutations_per_parent, config.mutation_strength
        )


class EvolutionalAgent(Agent):
    def __init__(self, config: EvolutionalAgentConfig) -> None:
        self.config = config
        self._policy_model = Model(
            LinearLayer(24, 64, activation=relu),
            LinearLayer(64, 32, activation=relu),
            LinearLayer(32, 4, activation=np.tanh),
        )

    def action(self, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self._policy_model(state)

    def train(self, env: gym.Env) -> List[float]:
        rewards = []
        sum_reward = 0
        parent_replaced = 0
        init_parent = deepcopy(self._policy_model)
        parent = Candidate(init_parent, self._play(env, init_parent))
        with trange(0, self.config.generations) as tgen:
            for generation in tgen:
                tgen.set_postfix(
                    generation=generation,
                    fitness=parent.fitness,
                )
                children: List[Candidate] = []
                for _ in range(self.config.mutations_per_parent):
                    child = self._mutate(parent.model)
                    fitness = self._play(env, child)
                    children.append(Candidate(child, fitness))

                fitnesses = [child.fitness for child in children]
                fittest_child = children[np.argmax(fitnesses)]
                if fittest_child.fitness > parent.fitness:
                    parent = fittest_child
                    parent_replaced += 1

                sum_reward = self._play(env, parent.model)

                rewards.append(sum_reward)

        self._policy_model = parent.model
        return rewards

    def _play(self, env: gym.Env, policy_model: Layer) -> float:
        state, done, sum_reward = env.reset(), False, 0
        while not done:
            action = policy_model(state)
            new_state, reward, done, _ = env.step(action)
            sum_reward += reward
            state = new_state

        return sum_reward

    def _mutate(self, model: Model) -> Model:
        base = deepcopy(model)
        for layer in base._layers:
            layer.W += np.random.normal(
                scale=self.config.mutation_strength, size=layer.W.shape
            )
            layer.b += np.random.normal(
                scale=self.config.mutation_strength, size=layer.b.shape
            )
        return base
