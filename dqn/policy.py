import torch
import numpy as np
import abc

class AbstractPolicy(abc.ABC):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
    
    @abc.abstractmethod
    def get_action(state, env, net: torch.nn, epsilon: float =0.0):
        pass

class PolicyEpsilongGreedy(AbstractPolicy):
    def __init__(self, device) -> None:
        super().__init__(device)

    def get_action(self, state, env, net: torch.nn, epsilon: float =0.0):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state = torch.tensor(
                np.array([state]), device=torch.device(self.device)
            ) 
            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
        return action