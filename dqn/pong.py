from dqn.dqn_base import ConcreteDQN
from dqn.dqn_memory import DQNWithMemory
from dqn.dqn_targetnn import DQNWithTargetNet
from dqn.neural_net import NeuralNetworkWithCNN
from dqn.memory import ExperienceReplay
import gymnasium as gym
import torch
import random
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("_mpl-gallery")


def preprocess_state(state, device):
    # print(state.dtype)
    state = np.expand_dims(state, axis=0)
    # print("Process state", state.shape)
    # state = state.transpose((2, 0, 1))
    # print(state.shape)
    state = torch.from_numpy(state)
    state = state.to(device, dtype=torch.float32)
    state = state.unsqueeze(1)
    return state


device = torch.device("cpu")
# device = torch.device("mps")

env_id = "PongNoFrameskip-v4"
# env_id = "ALE/Pong-v5"
# env = make_atari(env_id)
# env = wrap_deepmind(env, frame_stack=True)

env = gym.make(env_id, render_mode="human")
env = gym.wrappers.AtariPreprocessing(env)

seed_value = 23

torch.manual_seed(seed_value)
random.seed(seed_value)

learning_rate = 0.0001
num_episodes = 500
gamma = 0.99

hidden_layer = 512

report_interval = 20
number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

qnet_agent = ConcreteDQN(
    env,
    NeuralNetworkWithCNN(number_of_inputs, hidden_layer, number_of_outputs).to(device),
    learning_rate,
    device,
    gamma,
    preprocess_state_func=preprocess_state,
    egreedy=0.9,
    egreedy_final=0.01,
    egreedy_decay=10000,
    save_model_frequency=10000,
    resume_previous_training=True,
)

qnet_agent_with_memory = DQNWithMemory(
    qnet_agent,
    ExperienceReplay(100000),
    32,
)

qnet_agent_with_targetnn = DQNWithTargetNet(
    qnet_agent,
    NeuralNetworkWithCNN(number_of_inputs, hidden_layer, number_of_outputs).to(device),
    50000,
)

qnet_agent_with_memory_and_targetnn = DQNWithTargetNet(
    qnet_agent_with_memory,
    NeuralNetworkWithCNN(number_of_inputs, hidden_layer, number_of_outputs).to(device),
    50000,
)

qnet_agent_with_memory_and_dueling = DQNWithTargetNet(
    qnet_agent_with_memory,
    NeuralNetworkWithCNN(number_of_inputs, hidden_layer, number_of_outputs).to(device),
    50000,
)

qnet_agent_with_memory_and_targetnn.learn(num_episodes, seed_value, report_interval)
