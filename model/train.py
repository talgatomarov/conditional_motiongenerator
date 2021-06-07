from argparse import ArgumentParser
from MotionGeneratorDataModule import MotionGeneratorDataModule
from MotionGenerator import  MotionGenerator
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_top_k", type=int, default=1)

    parser = MotionGenerator.add_model_specific_args(parser)
    parser = MotionGeneratorDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = MotionGenerator(**vars(args))
    datamodule = MotionGeneratorDataModule.from_argparse_args(args, tokenizer=model.tokenizer)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=args.save_top_k,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, datamodule=datamodule)