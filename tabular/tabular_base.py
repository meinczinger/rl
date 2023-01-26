import gymnasium as gym
import abc
import torch


class TabularBase(abc.ABC):
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 0.3,
        epsilon_final: float = 0.01,
        epsilon_decay: float = 0.95,
        gamma: float = 0.9,
    ) -> None:
        self._epsilon = epsilon
        self._epsilon_final = epsilon_final
        self._epsilon_decay = epsilon_decay
        self._gamma = gamma
        self._env = env
        self.initialize()

    def initialize(self):
        self._q_values = torch.zeros(
            [self._env.observation_space.n, self._env.action_space.n]
        )

    @abc.abstractmethod
    def choose_action(self, state: int, epsilon: float) -> int:
        pass

    @abc.abstractmethod
    def step(self, state, new_state, action, reward, done) -> float:
        pass

    def test(self) -> list:
        return [0, 1]

    def learn(self, nr_of_episodes: int) -> tuple:
        self.initialize()
        steps_total = []
        scores_total = []

        epsilon = self._epsilon
        for _ in range(nr_of_episodes):
            step = 0
            score = 0
            state, _ = self._env.reset()

            while True:
                step += 1

                action = self.choose_action(state, epsilon)

                if epsilon > self._epsilon_final:
                    epsilon *= self._epsilon_decay

                new_state, reward, terminated, aborted, info = self._env.step(action)

                score += reward

                self._q_values[state, action] = self.step(
                    state, new_state, action, reward, terminated or aborted
                )

                state = new_state

                if terminated:
                    steps_total.append(step)
                    scores_total.append(score)
                    break

        return steps_total, scores_total
