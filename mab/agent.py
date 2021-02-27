import numpy as np
from enum import Enum


class RewardStrategy:
    """ Abstract class for variuos reward strategies used for simulations"""
    def __init__(self):
        pass

    def reward(self):
        raise NotImplementedError


class ConstantRewardStrategy(RewardStrategy):
    """ Reward strategy returning a constant reward """
    def __init__(self, reward):
        super(ConstantRewardStrategy, self).__init__()
        self.reward = reward

    def reward(self):
        return self.reward


class RandomRewardStrategy(RewardStrategy):
    """ Reward strategy returning a reward drawn from a standard distribution """
    def __init__(self, mean, std):
        """ Take mean and standard deviation """
        super(RandomRewardStrategy, self).__init__()
        self.mean = mean
        self.std = std

    def reward(self):
        """ Return a random number drawn from the standard distribution """
        return np.random.normal(self.mean, self.std)


class Arm:
    """ Simple class representing an arm """
    def __init__(self, reward_strategy):
        self.reward_strategy = reward_strategy

    def reward(self):
        """ Return the reward following the reward strategy used by the arm """
        return self.reward_strategy.reward()


class Environment:
    """ Class representing the environment (bandit), which essentially consists of a number of arms """
    def __init__(self, arm):
        self.arm = arm

    def get_nr_of_arms(self) -> int:
        """ Return the nr of arms """
        return len(self.arm)

    def reward(self, arm):
        """ Return the reward for a particular arm """
        return self.arm[arm].reward_strategy.reward()


class ActionSelector:
    def __init__(self):
        pass

    def select_action(self, q_values: [float], arm_count: [int], step: int) -> int:
        raise NotImplementedError


class EpsilonGreedyActionSelector:
    """
    With probability 1-epsilon, choose the action with the highest value (or randomly among them if
    there are more than one. Otherwise choose a random action to support exploration
    """
    def __init__(self, epsilon):
        super(EpsilonGreedyActionSelector, self).__init__()
        self.epsilon = epsilon

    def select_action(self, q_values: [float], arm_count: [int], step: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(q_values))
        else:
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))


class UpperConfidenceBoundActionSelector:
    """
    Choose the action with the highest upper confidence bound value
    """
    def __init__(self, c):
        super(UpperConfidenceBoundActionSelector, self).__init__()
        self.c = c

    def select_action(self, q_values: [float], arm_count: [int], step: int) -> int:
        return np.argmax(q_values + self.c*np.sqrt(np.log(step)/(arm_count+1.0)))


class ActionSelectorEnum(Enum):
    EPSILON_GREEDY = 0
    UPPER_CONFIDENCE_BOUND = 1


ACTION_SELECTOR_TYPE = {ActionSelectorEnum.EPSILON_GREEDY: EpsilonGreedyActionSelector,
                        ActionSelectorEnum.UPPER_CONFIDENCE_BOUND: UpperConfidenceBoundActionSelector}


class ActionSelectorFactory(object):
    @staticmethod
    def get_action_selector(action_selector_type: ACTION_SELECTOR_TYPE, **kwargs):
        try:
            return ACTION_SELECTOR_TYPE[action_selector_type](**kwargs)
        except Exception:
            raise Exception


class Agent:
    """ Class representing the agent """
    def __init__(self, environment: Environment, epsilon: float,
                 action_selector: ActionSelector, initial_value: float = 0):
        """
        environment - the environment the agent is going to interact with
        epsilon - the value of epsilon to control exploration
        initial_value - Optimistic initial value - to support early exploration
        """
        self.env = environment
        self.epsilon = epsilon
        self.action_selector = action_selector
        self.q_values = np.full(self.env.get_nr_of_arms(), initial_value)
        self.arm_counts = np.zeros(self.env.get_nr_of_arms())
        self.step = 1

    def agent_step(self) -> (int, float):
        """ Performs one step
        Returns:
            action - the action taken
            reward - the reward received
        """
        # Find the action with the highest value
        action = self.action_selector.select_action(self.q_values, self.arm_counts, self.step)
        # Get the reward from the environment
        reward = self.env.reward(action)
        self.arm_counts[action] += 1
        # Update the estimated action value
        self.q_values[action] = self.q_values[action] + \
                                (reward - self.q_values[action]) / self.arm_counts[action]
        # Keep the number of steps the agent made up-to-date
        self.step += 1

        return action, reward


class Simulation:
    """ Class to run a simulation """
    def __init__(self, nr_of_arms: int, epsilon: float, nr_of_steps: int,
                 action_selector: ActionSelector, initial_value: float = 0.0):
        """
        nr_of_arms - how many arms the environment will have
        epsilon - parameter to contorl exploration
        nr_of_steps - how many steps to simulate
        initial_value - Optimistic initial value - to support early exploration
        """
        self.nr_of_arms = nr_of_arms
        self.epsilon = epsilon
        self.nr_of_steps = nr_of_steps
        self.initial_value = initial_value
        self.action_selector = action_selector

    def simulate(self) -> [float]:
        """
        Simulates nr_of_steps steps
        :return:
        the average rewards collected during the simulation
        """
        arms = []
        # Create the necessary arms
        for i in range(self.nr_of_arms):
            arms.append(Arm(RandomRewardStrategy(np.random.normal(0, 1), 1)))
        # Create the environment
        env = Environment(arms)
        # Create the agent
        agent = Agent(env, self.epsilon, self.action_selector, self.initial_value)

        scores = [0]
        averages = []

        # Run the simulation
        for i in range(self.nr_of_steps):
            action, reward = agent.agent_step()
            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))

        return averages
