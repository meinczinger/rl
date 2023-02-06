from dqn.dqn_base import ConcreteDQN
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
    # state = state.transpose((2, 0, 1))
    return state
    # frame = np.expand_dims(frame, axis=0)
    # frame = frame.transpose((2,0,1))
    # frame = torch.from_numpy(frame)
    # frame = frame.to(device, dtype=torch.float32)
    # frame = frame.unsqueeze(1)
    
    # return frame


device = torch.device("cpu")
# device = torch.device("mps")

env_id = "PongNoFrameskip-v4"
# env_id = "ALE/Pong-v5"
# env = make_atari(env_id)
# env = wrap_deepmind(env, frame_stack=True)

env = gym.make(env_id, render_mode="human")
# env = gym.make(env_id)
# env = gym.wrappers.AtariPreprocessing(env, frame_skip=4)
env = gym.wrappers.AtariPreprocessing(env)

seed_value = 23

torch.manual_seed(seed_value)
random.seed(seed_value)

learning_rate = 0.0001
num_episodes = 500
gamma = 0.99

hidden_layer = 512

batch_size = 32
experience_memory_size = 100000

update_target_frequency = 2000

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
    egreedy=0.01,
    egreedy_final=0.01,
    egreedy_decay=10000,
    save_model_frequency=10000,
    resume_previous_training=True,
    experience_memory_size=experience_memory_size,
    batch_size=batch_size,
    target_net=NeuralNetworkWithCNN(number_of_inputs, hidden_layer, number_of_outputs).to(device),
    update_target_frequency=update_target_frequency
)

qnet_agent.learn(num_episodes, seed_value, report_interval)
