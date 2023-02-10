import random
from collections import deque
from torch.utils.data.dataset import IterableDataset
import torch
import numpy as np


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
        self.max_priority = 0.0

    def append(self, experience):
        super().append(experience)
        self.priorities.append(self.max_priority)

    def update(self, index, priority):
        if priority > self.max_priority:
            self.max_priority = priority
        self.priorities[index] = priority

    def sample(self, batch_size):
        prios = np.array(self.priorities, dtype=np.float64) + 1e-4
        prios = prios ** self.alpha
        probs = prios / prios.sum()
        
        weights = (self.__len__() * probs) ** -self.beta
        weights = weights / weights.max()

        idx = random.choice(range(self.__len__()), weights=probs, k=batch_size)
        sample = [(i, weights[i], *self.buffer[i]) for i in idx]
        return sample

class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=400):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience
            # yield [e.unsqueeze(0) for e in experience]
