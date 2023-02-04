import torch
from dqn.neural_net import NeuralNetwork
import gymnasium as gym
import abc
import math
import os


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
        file2save: str = "pong_save.pth",
        save_model_frequency: int = 10000,
        resume_previous_training: bool = False,
        preprocess_state_func=None,
    ):
        self.env = env
        self.neural_net = neural_net
        self.lr = lr
        self.device = device
        self.gamma = gamma
        self.egreedy = egreedy
        self.egreedy_final = egreedy_final
        self.egreedy_decay = egreedy_decay
        self.file2save = file2save
        self.save_model_frequency = save_model_frequency
        self.preprocess_state_func = preprocess_state_func

        if resume_previous_training and os.path.exists(self.file2save):
            self.neural_net.load_state_dict(self.load_model())

        self.loss_func = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            params=self.neural_net.parameters(), lr=self.lr
        )

        self.number_of_steps = 0

    def preprocess_state(self, state):
        if self.preprocess_state_func is not None:
            return self.preprocess_state_func(state, self.device)
        else:
            return torch.Tensor(state).to(self.device)

    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:
            with torch.no_grad():
                # print("State", state.shape)
                state = self.preprocess_state(state)
                # print("After preprocess", state.shape)
                state = torch.cat([state])
                # print("After cat", state.shape)
                action_from_nn = self.neural_net(state)
                # print(action_from_nn)
                action = torch.max(action_from_nn, 1)[1]
                action = action.item()
                # print("Action", action)
        else:
            action = self.env.action_space.sample()

        return action

    def prepare_tensors(self, *args):
        state, new_state, reward, action, done = args
        return (
            # self.preprocess_state(state),
            torch.Tensor(state).to(self.device),
            # self.preprocess_state(new_state),
            torch.Tensor(new_state).to(self.device),
            torch.Tensor([reward]).to(self.device),
            torch.LongTensor([action]).to(self.device),
            torch.Tensor([done]).to(self.device),
        )

    def predicted_value(self, state, new_state, reward, action, done):
        new_state_values = self.new_state_values(new_state)
        max_new_state_values = torch.max(new_state_values)
        target_value = reward + (1 - done) * self.gamma * max_new_state_values

        prepared_state = torch.cat([self.preprocess_state(state)])
        # print("predicted_value", prepared_state.shape)
        predicted_values = self.neural_net(prepared_state)
        # print(predicted_values)
        # print(action)
        predicted_value = predicted_values[0][action[0]]
        return predicted_value, target_value

    def optimize(self, predicted_value, target_value):
        if predicted_value is None:
            return

        # print("Predicted value", predicted_value)
        # print("Target value", target_value)
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
        if (self.number_of_steps > 0) and (
            self.number_of_steps % self.save_model_frequency == 0
        ):
            self.save_model(self.neural_net)

    def after_optimize(self):
        pass

    def new_state_values(self, new_state):
        pre_processed_state = torch.cat([self.preprocess_state(new_state)])

        # print("Before new state value", pre_processed_state.shape)
        return self.neural_net(pre_processed_state).detach()

    def reset_env(self, seed_value):
        if seed_value == 0:
            state, _ = self.env.reset()
        else:
            state, _ = self.env.reset(seed=seed_value)
        return state

    def load_model(self):
        print("Loading model")
        return torch.load(self.file2save)

    def save_model(self, model):
        print("Saving model")
        torch.save(model.state_dict(), self.file2save)

    def learn(
        self,
        nr_of_episodes: int,
        seed_value: int = 0,
        report_interval: int = 100,
        metrics_callback=None,
    ):
        rewards_total = []

        frames_total = 0

        for i_episode in range(nr_of_episodes):
            state = self.reset_env(seed_value)
            # print("--- State ---")
            # print(state.shape)
            # print(type(state))
            # print(state.dtype)

            score = 0
            done = False
            while not done:
                frames_total += 1

                epsilon = self.calculate_epsilon(frames_total)

                action = self.select_action(state, epsilon)

                new_state, reward, done = self.step(action)

                score += reward

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
                        state,
                        new_state,
                        reward_tensor,
                        action_tensor,
                        done_tensor,
                    )

                    self.optimize(predicted_value, target_value)

                    self.number_of_steps += 1

                    self.after_optimize()

                    # state = state.transpose((2, 0, 1))
                state = new_state

                if done:
                    print(i_episode, score)
                    rewards_total.append(score)

                    if (i_episode % report_interval == 0) and (
                        metrics_callback is not None
                    ):
                        metrics_callback(rewards_total)

        return rewards_total


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
        return self.decorated_dqn.learn(
            nr_of_episodes, seed_value, report_interval, metrics_callback
        )
