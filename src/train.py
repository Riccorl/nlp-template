from pathlib import Path
from typing import Optional

import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.pl_modules import BasePLModule


def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.train.seed)

    if conf.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{conf.train.pl_trainer.fast_dev_run}>. Forcing debugger configuration"
        )
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.gpus = 0
        conf.train.pl_trainer.precision = 32
        conf.data.datamodule.num_workers = 0
        # Switch wandb mode to offline to prevent online logging
        conf.logging.wandb_arg.mode = "offline"

    # data module declaration
    hydra.utils.log.info(f"Instantiating the Data Module")
    pl_data_module = hydra.utils.instantiate(conf.data.datamodule)

    # main module declaration
    hydra.utils.log.info(f"Instantiating the Model")
    pl_module: pl.LightningModule = hydra.utils.instantiate(
        conf.model, model=conf.model.args, train=conf.train, labels=pl_data_module.labels
    )

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(
            conf.train.early_stopping_callback
        )
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(
            conf.train.early_stopping_callback
        )
        callbacks_store.append(model_checkpoint_callback)

    logger: Optional[WandbLogger] = None
    if conf.logging.log:
        hydra.utils.log.info(f"Instantiating Wandb Logger")
        Path(conf.logging.wandb_arg.save_dir).mkdir(parents=True, exist_ok=True)
        logger = hydra.utils.instantiate(conf.logging.wandb_arg)
        logger.watch(pl_module, **conf.logging.watch)

    # trainer
    hydra.utils.log.info(f"Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, logger=logger
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
