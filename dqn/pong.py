from dqn.dqn_base import ConcreteDQN, DeepQLearning
from dqn.neural_net import NeuralNetworkWithCNN, NNLunarLanderDueling
from dqn.replay_buffer import ReplayBuffer
from dqn.environment import DQNEnvironment
from dqn.policy import PolicyEpsilongGreedy
from dqn.temperature import Temperature
import gymnasium as gym
import torch
import torch.nn.functional as F
import random
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

env = DQNEnvironment("PongNoFrameskip-v4", atari_game=True)

obs_shape = env.env.observation_space.shape
number_of_outputs = env.env.action_space.n
hidden_layer = 512

device = "mps"

dqn = DeepQLearning(
    env.env,
    policy=PolicyEpsilongGreedy(device),
    q_net=NeuralNetworkWithCNN(hidden_layer, obs_shape, number_of_outputs), loss_fn=F.mse_loss, optim=torch.optim.Adam, lr= 0.0001, batch_size=32, sync_rate=1, \
        double_dqn=False, capacity=10000, priority_buffer=False, epsilon=Temperature(1.0, 0.01, 15000)
)

match (device):
    case "cpu":
        trainer = Trainer(
            max_epochs=10_000,
            callbacks=[
                EarlyStopping(monitor="episode/Return", mode="max", stopping_threshold=18.0, patience=10000)
            ],
        )
    case "mps":
        trainer = Trainer(
            max_epochs=10_000,
            callbacks=[
                EarlyStopping(monitor="episode/Return", mode="max", stopping_threshold=18.0, patience=10000)
            ],
            accelerator="mps",
            devices=1,
        )



trainer.fit(dqn)