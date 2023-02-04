import torch
from dqn.dqn_base import DQNDecorator, AbstractDQN
from dqn.memory import ExperienceReplay
from dqn.neural_net import NeuralNetwork
import gymnasium as gym


class DQNWithMemory(DQNDecorator):
    def __init__(
        self,
        decorated_dqn: AbstractDQN,
        memory: ExperienceReplay,
        batch_size: int,
    ):
        super().__init__(decorated_dqn)

        self.memory = memory
        self.batch_size = batch_size

    def after_step(self, state, new_state, action, reward, done):
        self.memory.push(state, action, new_state, reward, 1 if done else 0)

    def can_optimize(self):
        return len(self.memory) >= self.batch_size

    def prepare_tensors(self, *args):
        state, new_state, reward, action, done = args

        state, action, new_state, reward, done = self.memory.sample(self.batch_size)

        state = [self.decorated_dqn.preprocess_state(frame) for frame in state]
        torch.cat(state)

        new_state = [self.decorated_dqn.preprocess_state(frame) for frame in new_state]
        torch.cat(new_state)

        return (
            # torch.Tensor(state).to(self.device),
            state,
            # torch.Tensor(new_state).to(self.device),
            new_state,
            torch.Tensor(reward).to(self.device),
            torch.LongTensor(action).to(self.device),
            torch.Tensor(done).to(self.device),
        )

    def predicted_value(self, state, new_state, reward, action, done):
        new_state_values = self.neural_net(new_state).detach()
        max_new_state_values = torch.max(new_state_values, 1)[0]
        target_value = reward + (1 - done) * self.gamma * max_new_state_values

        predicted_value = (
            self.neural_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        )

        return predicted_value, target_value
