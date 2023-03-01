from torch import nn
import torch
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
    def __init__(self, hidden_layer, number_of_inputs, number_of_outputs):
        super().__init__()
        # super(NeuralNetworkWithCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.advantage1 = nn.Linear(7 * 7 * 64, hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, number_of_outputs)

        self.value1 = nn.Linear(7 * 7 * 64, hidden_layer)
        self.value2 = nn.Linear(hidden_layer, 1)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x / 255

        # print("Input", x.shape)
        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)

        # print("Before", output_conv.shape)
        output_conv = output_conv.view(output_conv.size(0), -1)
        # print("After", output_conv.shape)

        output_advantage = self.advantage1(output_conv)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)

        output_value = self.value1(output_conv)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)

        output_final = output_value + output_advantage - output_advantage.mean()

        return output_final


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
        return self.net(x.float())


class NNLunarLanderDueling(nn.Module):
    def __init__(self, hidden_size, obs_size, n_actions) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Q(s, a) = V(s) + Adv(s, a)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.fc_adv = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.net(x.float())
        adv = self.fc_adv(x)
        value = self.fc_value(x)
        return value + adv - torch.mean(adv, dim=1, keepdim=True)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma) -> None:
        super(NoisyLinear, self).__init__()

        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_sigma = nn.Parameter(torch.empty(out_features))

        kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        kaiming_uniform_(self.w_sigma, a=math.sqrt(5))
        zeros_(self.b_mu)
        zeros_(self.b_sigma)

    def forward(self, x, sigma=0.5):
        if self.training:
            w_noise = torch.normal(0, sigma, size=self.w_mu.size()).to(device)
            b_noise = torch.normal(0, sigma, size=self.b_mu.size()).to(device)

            return F.linear(
                x,
                self.w_mu + self.w_sigma * w_noise,
                self.b_mu + self.b_sigma * b_noise,
            )
        else:
            return F.linear(x, self.w_mu, self.b_mu)
