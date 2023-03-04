from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.nn.init import kaiming_uniform_, zeros_


class NeuralNetwork(nn.Module):
    def __init__(self, number_of_inputs, hidden_layer, number_of_outputs):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, number_of_outputs)

        self.activation = nn.Tanh()

    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)

        return output2


class NeuralNetworkForDueling(nn.Module):
    def __init__(self, number_of_inputs, hidden_layer, number_of_outputs):
        super(NeuralNetworkForDueling, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer)

        self.advantage1 = nn.Linear(hidden_layer, hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, number_of_outputs)

        self.value1 = nn.Linear(hidden_layer, hidden_layer)
        self.value2 = nn.Linear(hidden_layer, 1)

        self.activation = nn.Tanh()

    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)

        output_advantage = self.advantage1(output1)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)

        output_value = self.value1(output1)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)

        output_final = output_value + output_advantage - output_advantage.mean()

        return output_final


class NeuralNetworkWithCNN(nn.Module):
    def __init__(self, hidden_layer, obs_shape, n_actions, sigma=None):
        super(NeuralNetworkWithCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_shape[0], out_channels=32, kernel_size=8, stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_layer)
            if not sigma
            else NoisyLinear(conv_out_size, hidden_layer, sigma=sigma),
            nn.ReLU(),
            nn.Linear(hidden_layer, n_actions)
            if not sigma
            else NoisyLinear(hidden_layer, n_actions, sigma=sigma),
        )

    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.Tensor(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, x):
        x = self.conv(x.to(torch.float32)).view(x.size()[0], -1)  # (batch_size, num_features)
        return self.fc(x)


class NeuralNetworkWithCNNDueling(nn.Module):
    def __init__(self, hidden_layer, obs_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_shape[0], out_channels=32, kernel_size=8, stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.head = nn.Sequential(
            nn.Linear(conv_out_size, hidden_layer),
            nn.ReLU(),
        )

        self.fc_advantage = nn.Linear(hidden_layer, n_actions)
        self.fc_value = nn.Linear(hidden_layer, 1)

    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.Tensor(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, x):
        x = self.conv(x.to(torch.float32)).view(x.size()[0], -1)  # (batch_size, num_features)
        x = self.head(x)
        adv = self.fc_advantage(x)
        value = self.fc_value(x)
        return value + adv - torch.mean(adv, dim=1, keepdim=True)


class NNLunarLander(nn.Module):
    def __init__(self, hidden_size, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.to(torch.float32))


class NNLunarLanderDueling(nn.Module):
    def __init__(self, hidden_size, obs_size, n_actions) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.fc_value = nn.Linear(hidden_size, 1)
        self.fc_adv = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.net(x.to(torch.float32))
        adv = self.fc_adv(x)
        value = self.fc_value(x)
        return value + adv - torch.mean(adv, dim=1, keepdim=True)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma) -> None:
        super(NoisyLinear, self).__init__()

        self._sigma = sigma
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_sigma = nn.Parameter(torch.empty(out_features))

        kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        kaiming_uniform_(self.w_sigma, a=math.sqrt(5))
        zeros_(self.b_mu)
        zeros_(self.b_sigma)

    def forward(self, x):
        if self.training:
            w_noise = torch.normal(0, self._sigma, size=self.w_mu.size()).to("mps")
            b_noise = torch.normal(0, self._sigma, size=self.b_mu.size()).to("mps")

            return F.linear(
                x,
                self.w_mu + self.w_sigma * w_noise,
                self.b_mu + self.b_sigma * b_noise,
            )
        else:
            return F.linear(x, self.w_mu, self.b_mu)
