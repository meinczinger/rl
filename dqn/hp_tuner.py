from dqn.dqn_base import DQNType, DQNFactory, DeepQLearning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from dqn.environment import DQNEnvironment
from dqn.policy import PolicyEpsilongGreedy
from dqn.neural_net import NeuralNetworkWithCNN
import torch.nn.functional as F
import torch
from dqn.temperature import Temperature

class HPTuner:
    def __init__(self, nr_of_studies, nr_of_epochs: int, logger, config) -> None:
        self.nr_of_studies = nr_of_studies
        self.nr_of_epochs = nr_of_epochs
        self.config = config
        self.logger = logger

    def trainer(self, config, num_epochs: int, checkpoint_dir, **kwargs):
        # save_dir = "/Users/meinczinger/src/github/rl/lightning_logs"
        # algo = DQNFactory.get_dqn(**kwargs, **config)
        env = DQNEnvironment("PongNoFrameskip-v4", atari_game=True)

        obs_shape = env.env.observation_space.shape
        number_of_outputs = env.env.action_space.n
        hidden_layer = 512
        device = 'cpu'

        algo = DeepQLearning(
            env.env,
            policy=PolicyEpsilongGreedy(device),
            q_net=NeuralNetworkWithCNN(hidden_layer, obs_shape, number_of_outputs), loss_fn=F.mse_loss, optim=torch.optim.Adam, gamma=0.99, lr= 0.0001, batch_size=32, sync_rate=1000, \
            double_dqn=False, capacity=100000, replay_initial=10000, priority_buffer=False, epsilon=Temperature(1.0, 0.02, 100000), **config)

        save_dir = "/Users/meinczinger/github/rl/lightning_logs/hp_tune"
        logger = TensorBoardLogger(save_dir=save_dir)
        trainer = Trainer(
            max_epochs=num_epochs,
            # If fractional GPUs passed in, convert to int.
            # gpus=math.ceil(num_gpus),
            logger=logger,
            enable_progress_bar=False,
            callbacks=[
                TuneReportCallback(
                    {"avg_return": "episode/avg_return"},
                    on="train_epoch_end",
                )
            ],
        )
        trainer.fit(algo)

    def tune(self, **kwargs):

        scheduler = ASHAScheduler(
            max_t=self.nr_of_epochs, grace_period=1000, reduction_factor=2
        )

        reporter = CLIReporter(
            parameter_columns=["frames_per_epoch", "samples_per_epoch"],
            metric_columns=["avg_return"],
        )

        train_fn_with_parameters = tune.with_parameters(
            self.trainer, num_epochs=self.nr_of_epochs, **kwargs
        )

        resources_per_trial = {"cpu": 1, "gpu": 0}

        tuner = tune.Tuner(
            tune.with_resources(
                train_fn_with_parameters, resources=resources_per_trial
            ),
            tune_config=tune.TuneConfig(
                metric="avg_return",
                mode="max",
                scheduler=scheduler,
                num_samples=self.nr_of_studies,
            ),
            run_config=air.RunConfig(
                # name="tune_dqn_asha",
                progress_reporter=reporter,
                checkpoint_config=air.CheckpointConfig(
                    # We haven't implemented checkpointing yet. See below!
                    checkpoint_at_end=False
                ),
            ),
            param_space=self.config,
        )
        results = tuner.fit()

        return results.get_best_result().config
