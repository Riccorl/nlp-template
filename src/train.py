from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console

from src.data.pl_data_modules import BasePLDataModule
from model.pl_modules import BasePLModule


def train(conf: omegaconf.DictConfig) -> None:
    # fancy logger
    console = Console()
    # reproducibility
    pl.seed_everything(conf.train.seed)
    console.log(f"Starting training for [bold cyan]{conf.train.model_name}[/bold cyan] model")
    if conf.train.pl_trainer.fast_dev_run:
        console.log(
            f"Debug mode {conf.train.pl_trainer.fast_dev_run}. Forcing debugger configuration"
        )
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.gpus = 0
        conf.train.pl_trainer.precision = 32
        conf.data.datamodule.num_workers = 0
        # Switch wandb to offline mode to prevent online logging
        conf.logging.wandb_arg.mode = "offline"

    # data module declaration
    console.log(f"Instantiating the Data Module")
    pl_data_module: BasePLDataModule = hydra.utils.instantiate(conf.data.datamodule)

    # main module declaration
    console.log(f"Instantiating the Model")
    pl_module: BasePLModule = hydra.utils.instantiate(conf.model)

    # callbacks declaration
    callbacks_store = [RichProgressBar()]

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
        console.log(f"Instantiating Wandb Logger")
        Path(conf.logging.wandb_arg.save_dir).mkdir(parents=True, exist_ok=True)
        logger = hydra.utils.instantiate(conf.logging.wandb_arg)
        logger.watch(pl_module, **conf.logging.watch)

    # trainer
    console.log(f"Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, logger=logger
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
