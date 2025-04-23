import argparse
import pytorch_lightning as pl
from cub_classification.model import CUBModel
from cub_classification.dataset import CUBDataModule
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import optuna


def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    classification_weight = trial.suggest_float("classification_weight", 0.1, 1.0)
    regression_weight = trial.suggest_float("regression_weight", 0.1, 1.0)

        
    wandb_logger = WandbLogger(
        project="CUB-Hypertuning",
        name=f'trial-{trial.number}',
        save_dir='hypertuning',
        log_model=True,
    )

    wandb_logger.experiment.config.update({
        "classification_weight": classification_weight,
        "regression_weight": regression_weight,
        "learning_rate": lr,
    })

    data_module  = CUBDataModule(
        data_dir=Path(args.data_dir),
        batch_size=4,
        transform=None
    )
    
    data_module.setup()

    model = CUBModel(
        num_classes=200,
        train_classification=True,
        train_regression=True,
        classification_weight=classification_weight,
        regression_weight=regression_weight,
        lr=lr,
    )

    early_stoping = EarlyStopping(
        monitor="val_combined_metric",
        patience=2,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[early_stoping],
    )

    trainer.fit(model, datamodule=data_module)

    wandb_logger.experiment.finish()

    return trainer.callback_metrics["val_combined_metric"].item()



if __name__=="__main__":
    # Do training

    parser = argparse.ArgumentParser(description="Train a CUB model")

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--number-of-trials", type=int, default=20)
    # parser.add_argument("--train-classification", type=bool, default=True)
    # parser.add_argument("--train-regression", type=bool, default=True)
    # parser.add_argument("--classification-weight", type=float, default=1.0)
    # parser.add_argument("--regression-weight", type=float, default=1.0)
    # parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    study = optuna.create_study(
        study_name="CUB-Class-and_Regr",
        direction="maximize",
        storage="sqlite:///cub_optuna_study.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.number_of_trials)

    
    # wandb_logger = WandbLogger(
    #     project="CUB-Regression-Classification",
    #     name=f'{args.classification_weight}-{args.regression_weight}-{args.lr}',
    #     save_dir='reports',
    #     log_model=True,
    # )

    # wandb_logger.experiment.config.update({
    #     "classification_weight": args.classification_weight,
    #     "regression_weight": args.regression_weight,
    #     "learning_rate": args.lr,
    # })
    
    # data_module  = CUBDataModule(
    #     data_dir=Path(args.data_dir),
    #     batch_size=4,
    #     transform=None
    # )
    
    # data_module.setup()

    # model = CUBModel(
    #     num_classes=200,
    #     train_classification=args.train_classification,
    #     train_regression=args.train_regression,
    #     classification_weight=args.classification_weight,
    #     regression_weight=args.regression_weight,
    #     lr=args.lr,
    # )

    # trainer = pl.Trainer(
    #     max_epochs=5,
    #     logger=wandb_logger
    # )

    # trainer.fit(model, datamodule=data_module)



#         x = self.pool(x)
