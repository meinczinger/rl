from torch import nn
import torch
import numpy as np

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
    def __init__(self, hidden_layer, obs_shape, n_actions):
        super().__init__()
        # super(NeuralNetworkWithCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=obs_shape[0], out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.head - nn.Sequential(
            nn.Linear(conv_out_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
        )

        self.fc_advantage = nn.Linear(hidden_layer, n_actions)
        self.fc_value = nn.Linear(hidden_layer, 1)

    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.Tensor(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, x):
        # x = x / 255
        x = self.conv(x.float()).view(x.size()[0], -1) # (batch_size, num_features)
        x = self.head(x)
        adv = self.fc_advantage(x)
        value = self.fc_value(x)
        return value + adv - np.mean(adv, dim=1, keepdims=True)


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
