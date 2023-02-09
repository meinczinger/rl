from dqn.policy import PolicyEpsilongGreedy
from dqn.dqn_base import DeepQLearning, DQNFactory, DQNType
from dqn.neural_net import NNLunarLander
from dqn.environment import DQNEnvironment
from dqn.hp_tuner import HPTuner
from ray import tune

# Parameters
device = "cpu"
hidden_size = 128

env = DQNEnvironment("LunarLander-v2")
policy = PolicyEpsilongGreedy(device)
q_net = NNLunarLander(hidden_size, env.observation_size(), env.number_of_actions())


hp = HPTuner(
    50,
    1000,
    config={"lr": tune.loguniform(1e-5, 1e-3), "gamma": tune.loguniform(0.6, 1.0)},
)
# hp.tune({"algo_type": DQNType.DQN, "env": env.env, "policy": policy, "q_net": q_net})
best_hp = hp.tune(algo_type=DQNType.DQN, env=env.env, policy=policy, q_net=q_net)

print("Best hyperparameters found were: ", best_hp)
