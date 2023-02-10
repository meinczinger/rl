import random
from collections import deque
from torch.utils.data.dataset import IterableDataset
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = 1.0
        self.beta = 0.5
        

class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=400):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience
            # yield [e.unsqueeze(0) for e in experience]
