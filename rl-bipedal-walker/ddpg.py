from copy import deepcopy
from dataclasses import dataclass
from typing import List

import gym
import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig
from torch.nn import Linear, MSELoss, ReLU, Sequential, Tanh
from torch.optim import Adam
from tqdm import trange

from agent import Agent
from core import polyak_update
from replay_buffer import Batch, ReplayBuffer


def q_loss(pred: torch.Tensor) -> torch.Tensor:
    return -torch.mean(pred)


@dataclass()
class DDPGAgentConfig:
    polyak: float
    discount: float
    batch_size: int
    std_dev: float
    steps: int
    replay_buffer_size: int

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "DDPGAgentConfig":
        return DDPGAgentConfig(
            config.polyak,
            config.discount,
            config.batch_size,
            config.std_dev,
            config.steps,
            config.replay_buffer_size,
        )


class DDPGAgent(Agent):
    def __init__(self, config: DDPGAgentConfig) -> None:
        self.config = config
        self._policy_model = Sequential(
            Linear(24, 128), ReLU(), Linear(128, 128), ReLU(), Linear(128, 4), Tanh()
        )
        self._q_model = Sequential(
            Linear(24 + 4, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, 1),
        )
        self._target_policy_model = deepcopy(self._policy_model)
        for p in self._target_policy_model.parameters():
            p.requires_grad = False

        self._target_q_model = deepcopy(self._q_model)
        for p in self._target_q_model.parameters():
            p.requires_grad = False

        self._q_loss = MSELoss()
        self._q_optim = Adam(params=self._q_model.parameters())
        self._policy_loss = q_loss
        self._policy_optim = Adam(params=self._policy_model.parameters())

    def action(self, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        input = torch.tensor(state).view(1, -1)
        return self._policy_model(input).detach().numpy().reshape(-1)

    def train(self, env: gym.Env) -> List[float]:
        buffer = ReplayBuffer(
            self.config.replay_buffer_size,
            env.observation_space.shape,
            env.action_space.shape,
        )
        rewards = []
        state, done, sum_reward = env.reset(), False, 0
        curr_episode = 1
        with trange(0, self.config.steps) as tr:
            tr.set_postfix(curr_episode=curr_episode, last_sum_reward=sum_reward)
            for _ in tr:
                if buffer.is_full():
                    action = self.action(state)
                    noise = np.random.normal(scale=self.config.std_dev, size=4)
                    action = np.clip(action + noise, -1.0, 1.0)
                else:
                    action = env.action_space.sample()

                new_state, reward, done, _ = env.step(action)
                sum_reward += reward
                buffer.add(state, action, reward, done, new_state)
                if buffer.is_full():
                    batch = buffer.sample(self.config.batch_size)
                    self._optimize_models(batch)
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

    def _optimize_models(self, batch: Batch) -> None:
        state, action, reward, done, next_state = batch

        state_t = torch.tensor(state)
        action_t = torch.tensor(action)
        next_state_t = torch.tensor(next_state)
        with torch.no_grad():
            next_state_action = self._state_action(
                next_state_t, self._target_policy_model(next_state_t)
            )
            target_q = (
                self._target_q_model(next_state_action).detach().numpy().reshape(-1)
            )
            target_t = torch.tensor(
                (reward + self.config.discount * (1.0 - done) * target_q).reshape(-1, 1)
            )
        state_action_q = self._state_action(state_t, action_t)

        self._q_optim.zero_grad()
        loss = self._q_loss(self._q_model(state_action_q), target_t)
        loss.backward()
        self._q_optim.step()

        self._policy_optim.zero_grad()
        state_action_p = self._state_action(state_t, self._policy_model(state_t))
        loss = self._policy_loss(self._q_model(state_action_p))
        loss.backward()
        self._policy_optim.step()

        polyak_update(self._policy_model, self._target_policy_model, self.config.polyak)
        polyak_update(self._q_model, self._target_q_model, self.config.polyak)

    def _state_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat((state, action), 1)
