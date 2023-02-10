import torch
import gymnasium as gym
import abc
import math
import os
from torch.utils.tensorboard import SummaryWriter
from dqn.neural_net import NeuralNetwork
from dqn.replay_buffer import ReplayBuffer

import copy
import gymnasium as gym
import torch
import random
import statistics
from enum import IntEnum

import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple
from IPython.display import HTML
from base64 import b64encode

from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW
from pytorch_lightning import LightningModule

from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit

from dqn.replay_buffer import ReplayBuffer, RLDataset
from dqn.neural_net import NNLunarLander
from dqn.epsilon import Epsilon


writer = SummaryWriter("runs/pong")


class ConcreteDQN:
    def __init__(
        self,
        env: gym.Env,
        neural_net: NeuralNetwork,
        lr: float,
        device,
        gamma: float,
        clip_error: bool = True,
        double_dqn: bool = True,
        target_net: NeuralNetwork = None,
        update_target_frequency: int = 1000,
        experience_memory_size: int = 1,
        batch_size: int = 32,
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
        self.clip_error = clip_error
        self.double_dqn = double_dqn
        self.egreedy = egreedy
        self.egreedy_final = egreedy_final
        self.egreedy_decay = egreedy_decay
        self.file2save = file2save
        self.save_model_frequency = save_model_frequency
        self.preprocess_state_func = preprocess_state_func

        if target_net is None:
            self.target_net = neural_net
            self.use_target_net = False
        else:
            self.target_net = target_net
            self.update_target_frequency = update_target_frequency
            self.use_target_net = True

        if resume_previous_training and os.path.exists(self.file2save):
            self.neural_net.load_state_dict(self.load_model())
            self.target_net.load_state_dict(self.load_model())

        self.experience_memory_size = experience_memory_size
        self.experience_replay = ReplayBuffer(self.experience_memory_size)
        self.batch_size = batch_size

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
                state = self.preprocess_state(state)
                state = torch.cat([state])
                action_from_nn = self.neural_net(state)
                action = torch.max(action_from_nn, 1)[1]
                action = action.item()
        else:
            action = self.env.action_space.sample()

        return action

    def optimize(self):
        if len(self.experience_replay) < self.batch_size:
            return

        state, action, new_state, reward, done = self.experience_replay.sample(
            self.batch_size
        )

        state = [self.preprocess_state(frame) for frame in state]
        state = torch.cat(state)

        new_state = [self.preprocess_state(frame) for frame in new_state]
        new_state = torch.cat(new_state)

        reward = torch.Tensor(reward).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        done = torch.Tensor(done).to(self.device)

        if self.double_dqn:
            new_state_values = self.neural_net(new_state).detach()
            max_new_state_indexes = torch.max(new_state_values, 1)[1]

            new_state_values = self.target_net(new_state).detach()
            max_new_state_values = new_state_values.gather(
                1, max_new_state_indexes.unsqueeze(1)
            ).squeeze(1)
        else:
            new_state_values = self.target_net(new_state).detach()
            max_new_state_values = torch.max(new_state_values)

        target_value = reward + (1 - done) * self.gamma * max_new_state_values

        predicted_value = (
            self.neural_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        )

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()

        loss.backward()

        if self.clip_error:
            for param in self.neural_net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def calculate_epsilon(self, steps_done):
        epsilon = self.egreedy_final + (self.egreedy - self.egreedy_final) * math.exp(
            -1.0 * steps_done / self.egreedy_decay
        )
        return epsilon

    def after_optimize(self):
        if (self.number_of_steps > 0) and (
            self.number_of_steps % self.save_model_frequency == 0
        ):
            self.save_model(self.neural_net)
        if self.use_target_net and (
            (self.number_of_steps % self.update_target_frequency) == 0
        ):
            self.target_net.load_state_dict(self.neural_net.state_dict())

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

        for i_episode in range(nr_of_episodes):
            state = self.reset_env(seed_value)
            # print("--- State ---")
            # print(state.shape)
            # print(type(state))
            # print(state.dtype)

            score = 0
            done = False
            while not done:
                epsilon = self.calculate_epsilon(self.number_of_steps)

                action = self.select_action(state, epsilon)

                new_state, reward, done, truncated, _ = self.env.step(action)

                done = done or truncated

                score += reward

                self.experience_replay.push(
                    state, action, new_state, reward, 1 if done else 0
                )

                self.optimize()

                self.number_of_steps += 1

                self.after_optimize()

                state = new_state

                if done:
                    print(i_episode, score, epsilon)
                    rewards_total.append(score)

                    if (i_episode % report_interval == 0) and (
                        metrics_callback is not None
                    ):
                        metrics_callback(rewards_total)

                    writer.add_scalar("reward", score, i_episode)
                    writer.add_scalar("epsilon", epsilon, i_episode)

        return rewards_total


class AbstractDQN(LightningModule):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)


class DeepQLearning(AbstractDQN):

    # Initialize.
    def __init__(
        self,
        env: gym.Env,
        policy,
        q_net=torch.nn,
        capacity=100_000,
        batch_size=256,
        lr=1e-3,
        gamma=0.99,
        loss_fn=F.smooth_l1_loss,
        optim=AdamW,
        epsilon: Epsilon = Epsilon(eps_start=1.0, eps_end=0.15, eps_last_episode=100),
        samples_per_epoch=1000,
        sync_rate=10,
        double_dqn: bool = True,
    ):

        super().__init__()
        self.env = env

        self.q_net = q_net

        self.target_q_net = copy.deepcopy(self.q_net)

        self.policy = policy
        self.buffer = ReplayBuffer(capacity=capacity)

        self.epsilon = epsilon

        self.save_hyperparameters()

        while len(self.buffer) < self.hparams.samples_per_epoch:
            # print(f"{len(self.buffer)} samples in experience buffer. Filling...")
            self.play_episode(epsilon=self.epsilon.eps_start)

    @torch.no_grad()
    def play_episode(self, policy=None, epsilon=0.0):
        # print("Playing episode")
        state, _ = self.env.reset()
        done = False

        while not done:
            if policy:
                # print("Getting best action")
                action = policy.get_action(state, self.env, self.q_net, epsilon=epsilon)
            else:
                action = self.env.action_space.sample()

            # print("Taking action", action)
            next_state, reward, done, truncated, info = self.env.step(action)

            done = done or truncated
            # print("After step", next_state)
            # print(reward, type(reward))
            # print(reward.dtype)
            if type(reward) == int:
                reward = np.float32(reward)
            if type(reward) == float:
                reward = np.float32(reward)

            exp = (
                state,
                action,
                reward.astype(np.float32),
                done,
                next_state,
            )
            self.buffer.append(exp)
            state = next_state

    # Forward.
    def forward(self, x):
        # print("Forward")
        return self.q_net(x)

    # Configure optimizers.
    def configure_optimizers(self):
        q_net_optimizer = self.hparams.optim(
            self.q_net.parameters(), lr=self.hparams.lr
        )
        return [q_net_optimizer]

    # Create dataloader.
    def train_dataloader(self):
        # print("train_dataloader")
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        # print("dataset created")
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.hparams.batch_size, num_workers=0
        )
        # print("dataloader created")
        return dataloader

    # def on_train_batch_start(self):
    #   print("Batch start")

    # Training step.
    def training_step(self, batch, batch_idx):
        # print("training_step")
        states, actions, rewards, dones, next_states = batch
        if states.shape[0] < self.hparams.batch_size:
            return

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)

        state_action_values = self.q_net(states).gather(1, actions)

        if not self.hparams.double_dqn:
            next_action_values, _ = self.target_q_net(next_states).max(
                dim=1, keepdim=True
            )
            next_action_values[dones] = 0.0
        else:
            with torch.no_grad():
                _, next_actions = self.q_net(next_states).max(dim=1, keepdim=True)
                next_action_values = self.target_q_net(next_states).gather(
                    1, next_actions
                )
                next_action_values[dones] = 0.0

        expected_state_action_values = rewards + self.hparams.gamma * next_action_values

        loss = self.hparams.loss_fn(state_action_values, expected_state_action_values)
        self.log("episode/Q-Error", loss)
        return loss

    # Training epoch end.
    def training_epoch_end(self, training_step_outputs):
        # print("epoch end")
        # epsilon = max(
        #     self.hparams.eps_end,
        #     self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode,
        # )
        epsilon = self.epsilon.calculate_epsilon(self.current_epoch)

        self.play_episode(policy=self.policy, epsilon=epsilon)
        self.log("episode/Return", self.env.return_queue[-1][0])

        returns = list(self.env.return_queue)[-100:]
        self.log("hp_metric", np.mean(returns))

        if self.current_epoch % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


class DQNType(IntEnum):
    DQN = 0


class DQNFactory:
    @staticmethod
    def get_dqn(algo_type: DQNType, **kwargs) -> AbstractDQN:
        match (algo_type):
            case DQNType.DQN:
                return DeepQLearning(**kwargs)
