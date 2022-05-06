import torch
from torch.nn import Module
from torch import Tensor


class PolicyBasedQModel(Module):
    def __init__(self, q_model: Module, policy_model) -> None:
        super().__init__()
        self._q_model = q_model
        self._policy_model = policy_model

    def forward(self, x: Tensor) -> Tensor:
        action = self._policy_model(x)
        state_action = torch.cat((x, action), 1)
        return self._q_model(state_action)
