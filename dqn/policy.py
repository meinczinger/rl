import torch
import numpy as np
import abc
from dqn.temperature import Temperature


class AbstractPolicy(abc.ABC):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    @abc.abstractmethod
    def get_action(self, state, env, net: torch.nn):
        pass


class PolicyEpsilongGreedy(AbstractPolicy):
    def __init__(self, device, temperature: Temperature) -> None:
        super().__init__(device)
        self._temperature = temperature
        self._epsilon = temperature.start

    def get_action(self, state, env, net: torch.nn):
        if np.random.random() < self._epsilon:
            action = env.action_space.sample()
        else:
            state = torch.tensor(np.array([state]), device=torch.device(self.device))
            q_values = net(state)
            _, action = torch.max(q_values, dim=-1)
            action = int(action.item())
        return action

    @property
    def epsilon(self):
        return self._epsilon

    def update(self, steps: int):
        self._epsilon = self._temperature.calculate_temperature(steps)


class PolicyGreedy(AbstractPolicy):
    def __init__(self, device) -> None:
        super().__init__(device)

    def get_action(self, state, env, net: torch.nn):
        state = torch.tensor(np.array([state]), device=torch.device(self.device))
        q_values = net(state)
        _, action = torch.max(q_values, dim=-1)
        action = int(action.item())
        return action
