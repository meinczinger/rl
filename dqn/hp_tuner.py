from dqn.dqn_base import DQNType, DQNFactory
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining


class HPTuner:
    def __init__(self, nr_of_studies, nr_of_epochs: int, config) -> None:
        self.nr_of_studies = nr_of_studies
        self.nr_of_epochs = nr_of_epochs
        self.config = config

    def trainer(self, config, num_epochs: int, checkpoint_dir, **kwargs):
        save_dir = "/Users/meinczinger/src/github/rl/lightning_logs"
        algo = DQNFactory.get_dqn(**kwargs, **config)
        trainer = Trainer(
            max_epochs=num_epochs,
            # If fractional GPUs passed in, convert to int.
            # gpus=math.ceil(num_gpus),
            logger=TensorBoardLogger(save_dir=save_dir),
            enable_progress_bar=False,
            callbacks=[
                TuneReportCallback(
                    {"hp_metric": "hp_metric"},
                    on="train_epoch_end",
                )
            ],
        )
        trainer.fit(algo)

    def tune(self, **kwargs):

        scheduler = ASHAScheduler(
            max_t=self.nr_of_epochs, grace_period=10, reduction_factor=2
        )

        reporter = CLIReporter(
            parameter_columns=["lr", "gamma"],
            metric_columns=["hp_metric"],
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
                metric="hp_metric",
                mode="max",
                scheduler=scheduler,
                num_samples=self.nr_of_studies,
            ),
            run_config=air.RunConfig(
                # name="tune_dqn_asha",
                # progress_reporter=reporter,
                checkpoint_config=air.CheckpointConfig(
                    # We haven't implemented checkpointing yet. See below!
                    checkpoint_at_end=False
                ),
            ),
            param_space=self.config,
        )
        results = tuner.fit()

        return results.get_best_result().config
