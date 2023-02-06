import copy
import gymnasium as gym
import torch
import random

import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple
from IPython.display import HTML
from base64 import b64encode

from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.callbacks import EarlyStopping

from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'mps'
# num_gpus = torch.cuda.device_count()
num_gpus = 0

class DQN(nn.Module):

  def __init__(self, hidden_size, obs_size, n_actions):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions)     
    )
  
  def forward(self, x):
    return self.net(x.float())

def epsilon_greedy(state, env, net, epsilon=0.0):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        # print("Taking action")
        # print(state)
        # print("_______________________")
        # print(state.dtype)
        state = torch.tensor([state]).to(device)
        q_values = net(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())
    return action  

class ReplayBuffer:

  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def __len__(self):
    return len(self.buffer)
  
  def append(self, experience):
    self.buffer.append(experience)
  
  def sample(self, batch_size):
    return random.sample(self.buffer, batch_size)

class RLDataset(IterableDataset):

  def __init__(self, buffer, sample_size=400):
    self.buffer = buffer
    self.sample_size = sample_size
  
  def __iter__(self):
    # return iter(self.buffer.sample(self.sample_size))
    # counter = 0
    # print(counter, len(self.buffer))
    for experience in self.buffer.sample(self.sample_size):      
      # print(experience, counter)
      # counter += 1
      yield experience

def create_environment(name):
  env = gym.make(name)
  env = TimeLimit(env, max_episode_steps=400)
  # env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
  env = RecordEpisodeStatistics(env)
  return env

env = create_environment('LunarLander-v2')

class DeepQLearning(LightningModule):

  # Initialize.
  def __init__(self, env_name, policy=epsilon_greedy, capacity=100_000, 
               batch_size=256, lr=1e-3, hidden_size=128, gamma=0.99, 
               loss_fn=F.smooth_l1_loss, optim=AdamW, eps_start=1.0, eps_end=0.15, 
               eps_last_episode=100, samples_per_epoch=1000, sync_rate=10):
    
    super().__init__()
    self.env = create_environment(env_name)

    obs_size = self.env.observation_space.shape[0]
    n_actions = self.env.action_space.n

    self.q_net = DQN(hidden_size, obs_size, n_actions)

    self.target_q_net = copy.deepcopy(self.q_net)

    self.policy = policy
    self.buffer = ReplayBuffer(capacity=capacity)

    self.save_hyperparameters()

    while len(self.buffer) < self.hparams.samples_per_epoch:
      # print(f"{len(self.buffer)} samples in experience buffer. Filling...")
      self.play_episode(epsilon=self.hparams.eps_start)
    
  @torch.no_grad()
  def play_episode(self, policy=None, epsilon=0.):
    # print("Playing episode")
    state, _ = self.env.reset()
    done = False

    while not done:
      if policy:
        action = policy(state, self.env, self.q_net, epsilon=epsilon)
      else:
        action = self.env.action_space.sample()
      
      next_state, reward, done, terminated, info = self.env.step(action)
      # print("After step", next_state)
      exp = (state, action, reward, done, next_state)
      self.buffer.append(exp)
      state = next_state

  # Forward.
  def forward(self, x):
    # print("Forward")
    return self.q_net(x)

  # Configure optimizers.
  def configure_optimizers(self):
    q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
    return [q_net_optimizer]

  # Create dataloader.
  def train_dataloader(self):
    # print("train_dataloader")
    dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
    # print("dataset created")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=self.hparams.batch_size,
        num_workers=0
    )
    # print("dataloader created")
    return dataloader

  # def on_train_batch_start(self):
  #   print("Batch start")

  # Training step.
  def training_step(self, batch, batch_idx):
    # print("training_step")
    states, actions, rewards, dones, next_states = batch
    actions = actions.unsqueeze(1)
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)

    state_action_values = self.q_net(states).gather(1, actions)
    
    next_action_values, _ = self.target_q_net(next_states).max(dim=1, keepdim=True)
    next_action_values[dones] = 0.0

    expected_state_action_values = rewards + self.hparams.gamma * next_action_values

    loss = self.hparams.loss_fn(state_action_values, expected_state_action_values)
    self.log('episode/Q-Error', loss)
    return loss


  # Training epoch end.
  def training_epoch_end(self, training_step_outputs):
    # print("epoch end")
    epsilon = max(
        self.hparams.eps_end,
        self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode
    )

    self.play_episode(policy=self.policy, epsilon=epsilon)
    self.log('episode/Return', self.env.return_queue[-1][0])

    if self.current_epoch % self.hparams.sync_rate == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())

algo = DeepQLearning('LunarLander-v2')

# trainer = Trainer(
#     gpus=num_gpus,
#     max_epochs=10_000,
#     callbacks=[EarlyStopping(monitor='episode/Return', mode='max', patience=500)]
# )

trainer = Trainer()

trainer.fit(algo)