from dqn.dqn_base import DeepQLearning, DQNFactory, DQNType
from dqn.neural_net import (
    NeuralNetworkWithCNN,
    NNLunarLanderDueling,
    NeuralNetworkWithCNNDueling,
)
from dqn.replay_buffer import ReplayBuffer
from dqn.environment import DQNEnvironment
from dqn.policy import PolicyEpsilongGreedy, PolicyGreedy
from dqn.temperature import Temperature
import gymnasium as gym
import torch
import torch.nn.functional as F
import random
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dqn.hp_tuner import HPTuner
from ray import tune
from pytorch_lightning.loggers import TensorBoardLogger


env = DQNEnvironment("PongNoFrameskip-v4", atari_game=True)

obs_shape = env.env.observation_space.shape
number_of_outputs = env.env.action_space.n
hidden_layer = 512

device = "mps"

seed = 27


def train():
    save_dir = "/Users/meinczinger/github/rl/"

    dqn = DeepQLearning(
        env.env,
        seed=seed,
        # policy=PolicyEpsilongGreedy(device, Temperature(1.0, 0.02, 100000)),
        policy=PolicyGreedy(device),
        q_net=NeuralNetworkWithCNN(
            hidden_layer, obs_shape, number_of_outputs, sigma=0.5
        ),
        loss_fn=F.mse_loss,
        optim=torch.optim.Adam,
        gamma=0.99,
        lr=0.0001,
        batch_size=128,
        sync_rate=1000,
        frames_per_epoch=50,
        double_dqn=False,
        capacity=50000,
        replay_initial=10000,
        samples_per_epoch=1600,
        priority_buffer=False,
    )

    logger = TensorBoardLogger(save_dir=save_dir)
    match (device):
        case "cpu":
            trainer = Trainer(
                max_epochs=100000,
                callbacks=[
                    EarlyStopping(
                        monitor="episode/avg_return",
                        mode="max",
                        stopping_threshold=18.0,
                        patience=10000,
                    )
                ],
                logger=logger,
            )
        case "mps":
            trainer = Trainer(
                max_epochs=100000,
                callbacks=[
                    EarlyStopping(
                        monitor="episode/avg_return",
                        mode="max",
                        stopping_threshold=18.0,
                        patience=10000,
                    )
                ],
                accelerator="mps",
                devices=1,
                logger=logger,
            )

    trainer.fit(dqn)


def tune_hp():
    env = DQNEnvironment("PongNoFrameskip-v4", atari_game=True)

    obs_shape = env.env.observation_space.shape
    number_of_outputs = env.env.action_space.n
    hidden_layer = 512

    device = "cpu"

    policy = PolicyEpsilongGreedy(device)
    q_net = NeuralNetworkWithCNN(hidden_layer, obs_shape, number_of_outputs)
    save_dir = "/Users/meinczinger/github/rl/lightning_logs/hp_tune"
    logger = TensorBoardLogger(save_dir=save_dir)

    hp = HPTuner(
        nr_of_studies=50,
        nr_of_epochs=3000,
        logger=logger,
        config={
            "frames_per_epoch": tune.lograndint(1, 1000),
            "samples_per_epoch": tune.lograndint(32, 16384),
        },
    )

    # hp.tune({"algo_type": DQNType.DQN, "env": env.env, "policy": policy, "q_net": q_net})
    # best_hp = hp.tune(algo_type=DQNType.DQN, env=env.env, policy=policy, q_net=q_net, loss_fn=F.mse_loss, optim=torch.optim.Adam, gamma=0.99, lr= 0.0001, batch_size=32, sync_rate=1000, \
    #     double_dqn=False, capacity=100000, replay_initial=10000, priority_buffer=False, epsilon=Temperature(1.0, 0.02, 100000))
    best_hp = hp.tune()
    print("Best hyperparameters found were: ", best_hp)


# tune_hp()
train()
