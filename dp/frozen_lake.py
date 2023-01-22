import gymnasium as gym
from enum import IntEnum
import numpy as np
from collections import Counter
import torch


class Moves(IntEnum):
    Left = 0
    Down = 1
    Right = 2
    Up = 3


class FrozenLake:
    def __init__(
        self, epsilon: float = 0.3, epsilon_final: float = 0.01, epsilon_decay: float = 0.95, gamma: float = 0.9, table_size: int = 4
    ) -> None:
        self._epsilon = epsilon
        self._epsilon_final = epsilon_final
        self._epsilon_decay = epsilon_decay
        self._gamma = gamma
        self._table_size = table_size
        self._env = self.create_env()
        self._q_values = torch.zeros(
            [self._env.observation_space.n, self._env.action_space.n]
        )
        self._visited_counts = Counter()

    def create_env(self):
        return gym.make("FrozenLake-v1", is_slippery=False)

    def learn(self):
        self._fvmc.learn(10)

    def allowed_actions(self, state: int) -> list:
        actions = [Moves.Right, Moves.Left, Moves.Down, Moves.Up]
        row = int(state / self._table_size)
        col = state % self._table_size
        if row == 0:
            actions.remove(Moves.Up)
        if row == (self._table_size - 1):
            actions.remove(Moves.Down)
        if col == 0:
            actions.remove(Moves.Left)
        if col == (self._table_size - 1):
            actions.remove(Moves.Right)
        return actions

    def generate_episode(self, nr_of_episodes: int) -> list:
        steps_total = []
        rewards_total = []

        epsilon = self._epsilon
        for _ in range(nr_of_episodes):
            step = 0
            state, _ = self._env.reset()


            while True:
                step += 1

                if epsilon < torch.rand(1)[0]:
                    values_with_noise = (
                        self._q_values[state]
                        + torch.randn(1, self._env.action_space.n) / 1000.0
                    )

                    action = int(torch.max(values_with_noise, 1)[1])
                else:
                    action = self._env.action_space.sample()

                if epsilon > self._epsilon_final:
                    epsilon *= self._epsilon_decay  

                new_state, reward, terminated, aborted, info = self._env.step(action)

                self._q_values[state, action] = reward + self._gamma * torch.max(
                    self._q_values[new_state]
                )

                state = new_state

                if terminated:
                    steps_total.append(step)
                    rewards_total.append(reward)
                    print("Episode finished after {0} steps".format(step))
                    break

        print(self._q_values)
        print("Average number of steps {}".format(sum(steps_total) / nr_of_episodes))
        print("Average number of steps for last 100 episodes {}".format(sum(steps_total[-100:]) / 100))
        print("Percentage of successful episodes {}".format(sum(rewards_total)/nr_of_episodes))

mc = FrozenLake(epsilon=0.7)
mc.generate_episode(1000)