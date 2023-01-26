from tabular.tabular_base import TabularBase
import gymnasium as gym
import torch


class DP(TabularBase):
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 0.7,
        epsilon_final: float = 0.01,
        epsilon_decay: float = 0.95,
        gamma: float = 0.9,
    ) -> None:
        super().__init__(env, epsilon, epsilon_final, epsilon_decay, gamma)

    def choose_action(self, state: int, epsilon: float) -> int:
        if epsilon < torch.rand(1)[0]:
            values_with_noise = (
                self._q_values[state]
                + torch.randn(1, self._env.action_space.n) / 1000.0
            )

            return int(torch.max(values_with_noise, 1)[1])
        else:
            return self._env.action_space.sample()

    def step(self, state, new_state, action, reward, done) -> float:
        return reward + self._gamma * torch.max(self._q_values[new_state])
