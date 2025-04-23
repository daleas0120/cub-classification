import argparse
import pytorch_lightning as pl
from cub_classification.model import CUBModel
from cub_classification.dataset import CUBDataModule
from pathlib import Path




if __name__=="__main__":
    # Do training

    parser = argparse.ArgumentParser(description="Train a CUB model")

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--train-classification", type=bool, default=True)
    parser.add_argument("--train-regression", type=bool, default=True)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)

    args = parser.parse_args()

    data_module  = CUBDataModule(
        data_dir=Path(args.data_dir),
        batch_size=4,
        transform=None
    )
    
    data_module.setup()

    model = CUBModel(
        num_classes=200,
        train_classification=args.train_classification,
        train_regression=args.train_regression,
        classification_weight=args.classification_weight,
        regression_weight=args.regression_weight
    )

    trainer = pl.Trainer(
        max_epochs=10,
    )

    trainer.fit(model, datamodule=data_module)

