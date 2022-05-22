from concurrent.futures import ProcessPoolExecutor, as_completed
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
        executor = ProcessPoolExecutor()
        rewards = []
        parent_replaced = 0
        init_parent = deepcopy(self._policy_model)
        parent = Candidate(init_parent, EvolutionalAgent._play(env, init_parent))
        with trange(0, self.config.generations) as tgen:
            for generation in tgen:
                tgen.set_postfix(
                    generation=generation,
                    fitness=parent.fitness,
                )

                children_f = [
                    executor.submit(
                        EvolutionalAgent._create_candidate,
                        env,
                        parent.model,
                        self.config.mutation_strength,
                    )
                    for _ in range(self.config.mutations_per_parent)
                ]
                children = [child_f.result() for child_f in as_completed(children_f)]

                fitnesses = [child.fitness for child in children]
                fittest_child = children[np.argmax(fitnesses)]
                if fittest_child.fitness > parent.fitness:
                    parent = fittest_child
                    parent_replaced += 1

                rewards.append(parent.fitness)

        self._policy_model = parent.model
        executor.shutdown()
        return rewards

    @staticmethod
    def _create_candidate(
        env: gym.Env, parent: Model, mutation_strength: float
    ) -> Candidate:
        env_c = deepcopy(env)
        child = deepcopy(parent)
        for layer in child._layers:
            layer.W += np.random.normal(scale=mutation_strength, size=layer.W.shape)
            layer.b += np.random.normal(scale=mutation_strength, size=layer.b.shape)
        fitness = EvolutionalAgent._play(env_c, child)
        return Candidate(child, fitness)

    @staticmethod
    def _play(env: gym.Env, policy_model: Layer) -> float:
        state, done, sum_reward = env.reset(), False, 0
        while not done:
            action = policy_model(state)
            new_state, reward, done, _ = env.step(action)
            sum_reward += reward
            state = new_state

        return sum_reward
