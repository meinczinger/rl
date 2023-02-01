import torch
from dqn.neural_net import NeuralNetwork
import gymnasium as gym
import abc
import math


class AbstractDQN(abc.ABC):
    @abc.abstractmethod
    def learn(
        self,
        nr_of_episodes: int,
        seed_value: int = 0,
        report_interval: int = 100,
        metrics_callback=None,
    ):
        pass


class ConcreteDQN(abc.ABC):
    def __init__(
        self,
        env: gym.Env,
        neural_net: NeuralNetwork,
        lr: float,
        device,
        gamma: float,
        egreedy: float = 0.9,
        egreedy_final: float = 0.01,
        egreedy_decay: float = 500,
    ):
        self.env = env
        self.neural_net = neural_net
        self.lr = lr
        self.device = device
        self.gamma = gamma
        self.egreedy = egreedy
        self.egreedy_final = egreedy_final
        self.egreedy_decay = egreedy_decay

        self.loss_func = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            params=self.neural_net.parameters(), lr=self.lr
        )

    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:
            with torch.no_grad():
                state = torch.Tensor(state).to(self.device)
                action_from_nn = self.neural_net(state)
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()
        else:
            action = self.env.action_space.sample()

        return action

    def prepare_tensors(self, *args):
        state, new_state, reward, action, done = args
        return (
            torch.Tensor(state).to(self.device),
            torch.Tensor(new_state).to(self.device),
            torch.Tensor([reward]).to(self.device),
            torch.LongTensor([action]).to(self.device),
            torch.Tensor([done]).to(self.device),
        )

    def predicted_value(self, state, new_state, reward, action, done):
        new_state_values = self.new_state_values(new_state)
        max_new_state_values = torch.max(new_state_values)
        target_value = reward + (1 - done) * self.gamma * max_new_state_values

        predicted_value = self.neural_net(state)[action]

        return predicted_value, target_value

    def optimize(self, predicted_value, target_value):
        if predicted_value is None:
            return

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    def calculate_epsilon(self, steps_done):
        epsilon = self.egreedy_final + (self.egreedy - self.egreedy_final) * math.exp(
            -1.0 * steps_done / self.egreedy_decay
        )
        return epsilon

    def step(self, action):
        new_state, reward, done, truncated, _ = self.env.step(action)
        return new_state, reward, done or truncated

    def can_optimize(self):
        return True

    def after_step(self, state, new_state, action, reward, done):
        """noting to be done for the base DQN version"""
        pass

    def after_optimize(self):
        """noting to be done for the base DQN version"""
        pass

    def new_state_values(self, new_state):
        return self.neural_net(new_state).detach()

    def reset_env(self, seed_value):
        if seed_value == 0:
            state, _ = self.env.reset()
        else:
            state, _ = self.env.reset(seed=seed_value)
        return state

    def learn(
        self,
        nr_of_episodes: int,
        seed_value: int = 0,
        report_interval: int = 100,
        metrics_callback=None,
    ):
        steps_total = []

        frames_total = 0

        for i_episode in range(nr_of_episodes):
            state = self.reset_env(seed_value)
            step = 0
            done = False
            while not done:
                step += 1
                frames_total += 1

                epsilon = self.calculate_epsilon(frames_total)

                action = self.select_action(state, epsilon)

                new_state, reward, done = self.step(action)

                self.after_step(state, new_state, action, reward, done)

                if self.can_optimize():
                    (
                        state_tensor,
                        new_state_tensor,
                        reward_tensor,
                        action_tensor,
                        done_tensor,
                    ) = self.prepare_tensors(state, new_state, reward, action, done)

                    predicted_value, target_value = self.predicted_value(
                        state_tensor,
                        new_state_tensor,
                        reward_tensor,
                        action_tensor,
                        done_tensor,
                    )

                    self.optimize(predicted_value, target_value)

                state = new_state

                if done:
                    steps_total.append(step)

                    if (i_episode % report_interval == 0) and (
                        metrics_callback is not None
                    ):
                        metrics_callback(steps_total)


class DQNDecorator(AbstractDQN):
    def __init__(self, decorated_dqn: AbstractDQN):
        self.decorated_dqn = decorated_dqn

    def learn(
        self,
        nr_of_episodes: int,
        seed_value: int = 0,
        report_interval: int = 100,
        metrics_callback=None,
    ):
        self.decorated_dqn.learn(
            nr_of_episodes, seed_value, report_interval, metrics_callback
        )
