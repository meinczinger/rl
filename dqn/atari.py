
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from dqn.policy import PolicyEpsilongGreedy
from dqn.dqn_base import DeepQLearning
from dqn.neural_net import NNLunarLander
from dqn.environment import DQNEnvironment
import optuna

# Parameters
device = "cpu"
hidden_size = 128

env = DQNEnvironment("LunarLander-v2")

def objective(trial:optuna.trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)

    algo = DeepQLearning(env=env.env, policy=PolicyEpsilongGreedy(device), q_net=NNLunarLander(hidden_size, env.observation_size(), env.number_of_actions()), lr=lr, gamma=gamma)    

# algo = DeepQLearning(env=env.env, policy=PolicyEpsilongGreedy(device), q_net=NNLunarLander(hidden_size, env.observation_size(), env.number_of_actions()))

    callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor='hp_metric')

    match(device):
        case 'cpu':
            trainer = Trainer(
                max_epochs=1_000,
                callbacks=[callback],
            )
        case 'mps':
            trainer = Trainer(
                max_epochs=1_000,
                callbacks=[callback],
                accelerator="mps", devices=1
            )

    hyperparameters = dict(lr=lr, gamma=gamma)

    trainer.logger.log_hyperparams(hyperparameters)
    
    trainer.fit(algo)

    return trainer.callback_metrics['hp_metric'].item()

pruner = optuna.pruners.SuccessiveHalvingPruner()
study = optuna.create_study(direction="maximize", pruner=pruner)

study.optimize(objective, n_trials=20)