import torch
from dqn.dqn_base import DQNDecorator, AbstractDQN
from dqn.memory import ExperienceReplay
from dqn.neural_net import NeuralNetwork
import gymnasium as gym


class DQNWithTargetNet(DQNDecorator):
    def __init__(self, decorated_dqn: AbstractDQN, target_net: NeuralNetwork):
        super().__init__(decorated_dqn)

        self.target_net = target_net

        self.update_target_counter = 0


def after_optimize(self):
    if self.clip_error:
        for param in self.neural_net.parameters():
            param.grad.data.clamp_(-1, 1)

    self.optimizer.step()

    if (self.update_target_counter % self.update_target_frequency) == 0:
        self.target_net.load_state_dict(self.neural_net.state_dict())

    self.update_target_counter += 1


def new_state_values(self, new_state):
    return self.target_net(new_state).detach()
