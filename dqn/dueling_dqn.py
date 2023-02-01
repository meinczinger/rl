import torch
from dqn.memory import ExperienceReplay
from dqn.neural_net import NeuralNetworkForDueling

import gymnasium as gym


class DuelingDQN:
    def __init__(
        self,
        env: gym.Env,
        neural_net: NeuralNetworkForDueling,
        target_net: NeuralNetworkForDueling,
        memory: ExperienceReplay,
        lr: float,
        batch_size: int,
        device,
        gamma: float,
        clip_error: bool,
        update_target_frequency: int,
    ):
        self.env = env
        self.memory = memory
        self.neural_net = neural_net
        self.target_net = target_net
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.clip_error = clip_error
        self.update_target_frequency = update_target_frequency

        self.loss_func = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            params=self.neural_net.parameters(), lr=self.lr
        )

        self.update_target_counter = 0

    def select_action(self, state, epsilon):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:

            with torch.no_grad():

                state = torch.Tensor(state).to(self.device)
                action_from_nn = self.neural_net(state)
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()
        else:
            action = self.env.action_space.sample()

        return action

    def optimize(self):

        if len(self.memory) < self.batch_size:
            return

        state, action, new_state, reward, done = self.memory.sample(self.batch_size)

        state = torch.Tensor(state).to(self.device)
        new_state = torch.Tensor(new_state).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        done = torch.Tensor(done).to(self.device)

        new_state_indexes = self.neural_net(new_state).detach()
        max_new_state_indexes = torch.max(new_state_indexes, 1)[1]

        new_state_values = self.target_net(new_state).detach()
        max_new_state_values = new_state_values.gather(
            1, max_new_state_indexes.unsqueeze(1)
        ).squeeze(1)

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

        if (self.update_target_counter % self.update_target_frequency) == 0:
            self.target_net.load_state_dict(self.neural_net.state_dict())

        self.update_target_counter += 1
