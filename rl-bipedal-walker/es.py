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
    model: Model
    fitness: float
    mut_W: List[npt.NDArray[np.float32]]
    mut_b: List[npt.NDArray[np.float32]]


@dataclass()
class Parent:
    model: Model
    fitness: float


@dataclass()
class ESAgentConfig:
    generations: int
    mutations_per_parent: int
    mutation_strength: float
    learning_rate: float
    eval_parent_after_steps: int

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "ESAgentConfig":
        return ESAgentConfig(
            config.generations,
            config.mutations_per_parent,
            config.mutation_strength,
            config.learning_rate,
            config.eval_parent_after_steps,
        )


class ESAgent(Agent):
    def __init__(self, config: ESAgentConfig) -> None:
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
        parent = Parent(self._policy_model, ESAgent._play(env, self._policy_model))
        alpha = self.config.learning_rate / (
            self.config.mutation_strength**2 * self.config.generations
        )
        with trange(0, self.config.generations) as tgen:
            for generation in tgen:
                tgen.set_postfix(
                    generation=generation,
                    fitness=parent.fitness,
                )

                children_f = [
                    executor.submit(
                        ESAgent._create_candidate,
                        env,
                        parent.model,
                        self.config.mutation_strength,
                    )
                    for _ in range(self.config.mutations_per_parent)
                ]
                children = [child_f.result() for child_f in as_completed(children_f)]

                fitnesses = [child.fitness for child in children]
                mean_fitness = np.mean(fitnesses)
                std_fitness = np.std(fitnesses)
                weights = [
                    (fitness - mean_fitness) / std_fitness for fitness in fitnesses
                ]
                for id, layer in enumerate(parent.model._layers):
                    mut_W = np.array([child.mut_W[id] for child in children])
                    mut_b = np.array([child.mut_b[id] for child in children])
                    mut_W_mean = np.dot(mut_W.T, weights)
                    mut_b_mean = np.dot(mut_b.T, weights)
                    layer.W += alpha * mut_W_mean.T
                    layer.b += alpha * mut_b_mean.T

                if generation % self.config.eval_parent_after_steps == 0:
                    fitness = self._play(env, parent.model)
                    parent.fitness = fitness
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
        mut_W = []
        mut_b = []
        for layer in child._layers:
            m_w = np.random.normal(scale=mutation_strength, size=layer.W.shape)
            m_b = np.random.normal(scale=mutation_strength, size=layer.b.shape)
            mut_W.append(m_w)
            mut_b.append(m_b)
            layer.W += m_w
            layer.b += m_b
        fitness = ESAgent._play(env_c, child)
        return Candidate(child, fitness, mut_W, mut_b)

    @staticmethod
    def _play(env: gym.Env, policy_model: Layer) -> float:
        state, done, sum_reward = env.reset(), False, 0
        while not done:
            action = policy_model(state)
            new_state, reward, done, _ = env.step(action)
            sum_reward += reward
            state = new_state

        return sum_reward
