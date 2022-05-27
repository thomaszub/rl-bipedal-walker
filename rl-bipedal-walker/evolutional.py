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


@dataclass()
class Candidate:
    model: Model
    fitness: float


@dataclass()
class EvolutionalAgentConfig:
    generations: int
    children_per_parent: int
    mutation_strength: float
    num_parents: int
    nodes: List[int]

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "EvolutionalAgentConfig":
        return EvolutionalAgentConfig(
            config.generations,
            config.children_per_parent,
            config.mutation_strength,
            config.num_parents,
            config.nodes,
        )


class EvolutionalAgent(Agent):
    def __init__(self, config: EvolutionalAgentConfig) -> None:
        self.config = config
        self._policy_model = self._create_policy_model()

    def action(self, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self._policy_model(state)

    def train(self, env: gym.Env) -> List[float]:
        executor = ProcessPoolExecutor()
        rewards = []
        parents = [
            EvolutionalAgent._create_candidate(env, self._create_policy_model())
            for _ in range(self.config.num_parents)
        ]
        with trange(0, self.config.generations) as tgen:
            for generation in tgen:
                env.reset()
                best_parent_fitness = EvolutionalAgent._best_candidate(parents).fitness
                rewards.append(best_parent_fitness)
                tgen.set_postfix(
                    generation=generation,
                    fitness=best_parent_fitness,
                )

                candidates_f = [
                    executor.submit(
                        EvolutionalAgent._create_candidate,
                        env,
                        parent.model,
                        self.config.mutation_strength,
                    )
                    for parent in parents
                    for _ in range(self.config.children_per_parent)
                ]
                candidates_f.extend(
                    [
                        executor.submit(
                            EvolutionalAgent._create_candidate,
                            env,
                            parent.model,
                        )
                        for parent in parents
                    ]
                )
                candidates = [
                    candidate.result() for candidate in as_completed(candidates_f)
                ]
                candidates.sort(key=lambda x: x.fitness, reverse=True)
                parents = candidates[0 : self.config.num_parents]

        self._policy_model = EvolutionalAgent._best_candidate(parents).model
        executor.shutdown()
        return rewards

    @staticmethod
    def _create_candidate(
        env: gym.Env, parent: Model, mutation_strength: float = 0.0
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

    def _create_policy_model(self) -> Model:
        nodes = self.config.nodes

        layers = []
        layers.append(LinearLayer(24, nodes[0], activation=relu))

        for id in range(0, len(self.config.nodes) - 1):
            layers.append(LinearLayer(nodes[id], nodes[id + 1], activation=relu))

        layers.append(LinearLayer(nodes[-1], 4, activation=np.tanh))

        return Model(*layers)

    def _best_candidate(candidates: List[Candidate]) -> Candidate:
        id = np.argmax([candidate.fitness for candidate in candidates])
        return candidates[id]
