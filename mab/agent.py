import numpy as np


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


class Agent:
    """ Class representing the agent """
    def __init__(self, environment: Environment, epsilon: float):
        """
        environment - the environment the agent is going to interact with
        epsilon - the value of epsilon to control exploration
        """
        self.env = environment
        self.epsilon = epsilon
        self.q_values = np.zeros(self.env.get_nr_of_arms())
        self.arm_counts = np.zeros(self.env.get_nr_of_arms())

    def choose_action(self) -> int:
        """
        Chooses the best action, in case of a tie, choose randomly.
        Use epsilon to decide when to explore (choose a random action)
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.q_values))
        else:
            return np.random.choice(np.flatnonzero(self.q_values == self.q_values.max()))

    def agent_step(self) -> (int, float):
        """ Performs one step
        Returns:
            action - the action taken
            reward - the reward received
        """
        # Find the action with the highest value
        action = self.choose_action()
        # Get the reward from the environment
        reward = self.env.reward(action)
        self.arm_counts[action] += 1
        # Update the estimated action value
        self.q_values[action] = self.q_values[action] + \
                                (reward - self.q_values[action]) / self.arm_counts[action]
        return action, reward


class Simulation:
    """ Class to run a simulation """
    def __init__(self, nr_of_arms: int, epsilon: float, nr_of_steps: int):
        """
        nr_of_arms - how many arms the environment will have
        epsilon - parameter to contorl exploration
        nr_of_steps - how many steps to simulate
        """
        self.nr_of_arms = nr_of_arms
        self.epsilon = epsilon
        self.nr_of_steps = nr_of_steps

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
        agent = Agent(env, self.epsilon)

        scores = [0]
        averages = []

        # Run the simulation
        for i in range(self.nr_of_steps):
            action, reward = agent.agent_step()
            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))

        return averages
