import torch
import gymnasium as gym
import abc
import math
import os
from pytorch_lightning.loggers import TensorBoardLogger
from dqn.neural_net import NeuralNetwork
from dqn.replay_buffer import ReplayBuffer, PriorityReplayBuffer

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
from dqn.temperature import Temperature
from dqn.policy import PolicyEpsilongGreedy, PolicyGreedy

import time


class AbstractDQN(LightningModule):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)


class DeepQLearning(AbstractDQN):
    # Initialize.
    def __init__(
        self,
        env: gym.Env,
        policy,
        seed,
        q_net=torch.nn,
        capacity=100_000,
        frames_per_epoch: int = 1,
        priority_buffer: bool = False,
        batch_size=256,
        lr=1e-3,
        gamma=0.99,
        loss_fn=F.smooth_l1_loss,
        optim=AdamW,
        replay_initial=1000,
        samples_per_epoch=1000,
        sync_rate=10,
        double_dqn: bool = True,
        moving_average: int = 25,
        alpha: Temperature = Temperature(start=0.5, end=0.0, last_episode=100),
        beta: Temperature = Temperature(start=0.4, end=1.0, last_episode=100),
        n_steps: int = 1
    ):
        super().__init__()

        self.env = env

        self.q_net = q_net

        self.target_q_net = copy.deepcopy(self.q_net)

        self.policy = policy

        if priority_buffer:
            self.buffer = PriorityReplayBuffer(capacity=capacity)
        else:
            self.buffer = ReplayBuffer(capacity=capacity)

        self.alpha = alpha
        self.beta = beta

        self.start_time = time.time()

        self.frames = 0

        self.epochs = 0

        # self.logger = TensorBoardLogger('lightning_logs')

        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=["q_net"])

        self.state, _ = self.env.reset(seed=self.hparams.seed)

        # while len(self.buffer) < self.hparams.replay_initial:
        # print(f"{len(self.buffer)} samples in experience buffer. Filling...")
        self.play_epoch(nr_of_steps=self.hparams.replay_initial)

    @torch.no_grad()
    def play_epoch(self, policy=None, nr_of_steps: int = 1):
        transitions = []

        for _ in range(nr_of_steps):
            if policy:
                # print("Getting best action")
                action = policy.get_action(self.state, self.env, self.q_net)
            else:
                action = self.env.action_space.sample()

            self.frames += 1

            if self.frames % self.hparams.sync_rate == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # print("Taking action", action)
            next_state, reward, done, truncated, info = self.env.step(action)

            done = done or truncated
            # print("After step", next_state)
            # print(reward, type(reward))
            # print(reward.dtype)
            if type(reward) in [int, float]:
                reward = np.float32(reward)

            exp = (
                self.state,
                action,
                reward.astype(np.float32),
                done,
                next_state,
            )
            if self.hparams.n_steps > 1:
                transitions.append(exp)
            else:
                self.buffer.append(exp)

            self.state = next_state

            if done:
                self.state, _ = self.env.reset(seed=self.hparams.seed)
                self.epochs += 1

        if self.hparams.n_steps > 1:
            for i, (s, a, r, d, sn) in enumerate(transitions):
                batch = transitions[i:i + self.hparams.n_steps]
                ret = np.float32(sum([t[2] * self.hparams.gamma**j for j, t in enumerate(batch)]))
                _, _, _, ld, ls = batch[-1]
                self.buffer.append((s, a, ret, ld, ls))


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
        if not self.hparams.priority_buffer:
            states, actions, returns, dones, next_states = batch
        else:
            indices, weights, states, actions, returns, dones, next_states = batch
            weights = weights.unsqueeze(1)
        # if states.shape[0] < self.hparams.batch_size:
        #     return 0

        actions = actions.unsqueeze(1)
        returns = returns.unsqueeze(1)
        dones = dones.unsqueeze(1)
        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)

        state_action_values = self.q_net(states).gather(1, actions)

        if not self.hparams.double_dqn:
            with torch.no_grad():
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

        expected_state_action_values = returns + self.hparams.gamma**self.hparams.n_steps * next_action_values

        if self.hparams.priority_buffer:
            td_errors = (
                (state_action_values - expected_state_action_values).abs().detach()
            )

            for idx, e in zip(indices, td_errors):
                self.buffer.update(idx, e.item())

            loss = weights * self.hparams.loss_fn(
                state_action_values, expected_state_action_values, reduction="none"
            )
            loss = loss.mean()

        else:
            loss = self.hparams.loss_fn(
                state_action_values, expected_state_action_values
            )

        self.log("episode/Q-Error", loss)
        return loss

    # Training epoch end.
    def training_epoch_end(self, training_step_outputs):
        if type(self.policy) == PolicyEpsilongGreedy:
            self.policy.update(self.frames)

        if self.hparams.priority_buffer:
            alpha = self.alpha.calculate_temperature(self.current_epoch)
            beta = self.beta.calculate_temperature(self.current_epoch, False)
            self.buffer.alpha = alpha
            self.buffer.beta = beta

        # self.play_episode(policy=self.policy, epsilon=epsilon)
        self.play_epoch(
            policy=self.policy,
            nr_of_steps=self.hparams.frames_per_epoch,
        )
        # if done:
        self.log("episode/Return", self.env.return_queue[-1][0])

        returns = list(self.env.return_queue)[-self.hparams.moving_average :]
        self.log("episode/avg_return", np.mean(returns))

        steps_per_episode = list(self.env.length_queue)[-self.hparams.moving_average :]
        self.log("episode/episode_length", np.mean(steps_per_episode, dtype=np.float32))

        self.log(
            "episode/frames_per_sec", self.frames / (time.time() - self.start_time)
        )

        self.log("episode/frames", self.frames)

        if type(self.policy) == PolicyEpsilongGreedy:
            self.log("episode/epsilon", self.policy.epsilon)

        self.log("episode/epochs", self.epochs)


class DQNType(IntEnum):
    DQN = 0


class DQNFactory:
    @staticmethod
    def get_dqn(algo_type: DQNType, **kwargs) -> AbstractDQN:
        if algo_type == DQNType.DQN:
            return DeepQLearning(**kwargs)
