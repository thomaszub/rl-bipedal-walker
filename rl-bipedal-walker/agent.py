from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from gym import Space
from omegaconf import DictConfig
from torch.nn import Linear, MSELoss, ReLU, Sequential, Tanh
from torch.optim import Adam

from model import PolicyBasedQModel
from replay_buffer import ReplayBuffer


def q_loss(pred: torch.Tensor) -> torch.Tensor:
    return -torch.mean(pred)


@dataclass()
class DDPGAgentConfig:
    polyak: float
    discount: float
    batch_size: int

    @staticmethod
    def fromDictConfig(config: DictConfig) -> "DDPGAgentConfig":
        return DDPGAgentConfig(config.polyak, config.discount, config.batch_size)


class DDPGAgent:
    def __init__(
        self,
        config: DDPGAgentConfig,
        action_space: Space[np.ndarray],
        replay_buffer: ReplayBuffer,
    ) -> None:
        self.config = config
        self._action_space = action_space
        self._replay_buffer = replay_buffer
        self._policy_model = Sequential(Linear(24, 16), ReLU(), Linear(16, 4), Tanh())
        self._q_model = Sequential(
            # TODO Check how to insert policy model
            Linear(24 + 4, 16),
            ReLU(),
            Linear(16, 1),
        )
        self._target_policy_model = deepcopy(self._policy_model)
        self._target_q_model = deepcopy(self._q_model)
        self._target_pq_model = PolicyBasedQModel(
            self._target_q_model, self._target_policy_model
        )
        self._train_mode = True
        self._q_loss = MSELoss()
        self._q_optim = Adam(params=self._q_model.parameters())
        self._policy_loss = q_loss
        self._policy_optim = Adam(params=self._policy_model.parameters())

    def train_mode(self, on: bool) -> None:
        self._train_mode = on

    def action(self, state: np.ndarray) -> np.ndarray:
        input = torch.tensor(state).float().view(1, -1)
        action = (
            self._policy_model(input).detach().numpy().reshape(-1)
        )  # TODO Add random noice
        self._state = deepcopy(state)
        self._action = deepcopy(action)
        return action

    def update(self, new_state: np.ndarray, reward: float, done: bool) -> None:
        self._replay_buffer.add(self._state, self._action, reward, done, new_state)

        # if not self._replay_buffer.is_full():
        #    return
        # TODO When to start learning
        self._train()

    def _train(self) -> None:
        state, action, reward, done, next_state = self._replay_buffer.sample(
            self.config.batch_size
        )

        input = torch.tensor(next_state).float()
        target_q = self._target_pq_model(input).detach().numpy().reshape(-1)
        target = (reward + self.config.discount * (1.0 - done) * target_q).reshape(
            -1, 1
        )
        state_action = torch.tensor(np.concatenate((state, action), 1)).float()

        loss = self._q_loss(self._q_model(state_action), torch.tensor(target).float())
        self._q_optim.zero_grad()
        loss.backward()
        self._q_optim.step()

        state_t = torch.tensor(state).float()
        state_action = torch.cat((state_t, self._policy_model(state_t)), 1)
        loss = self._policy_loss(self._q_model(state_action))
        self._policy_optim.zero_grad()
        loss.backward()
        self._policy_optim.step()

        self._update_target_models()

    @torch.no_grad()
    def _update_target_models(self) -> None:
        polyak = self.config.polyak

        for name, p in self._target_policy_model.named_parameters():
            p_update = polyak * p + (1.0 - polyak) * self._policy_model.get_parameter(
                name
            )
            p.copy_(p_update)

        for name, p in self._target_q_model.named_parameters():
            p_update = polyak * p + (1.0 - polyak) * self._q_model.get_parameter(name)
            p.copy_(p_update)
