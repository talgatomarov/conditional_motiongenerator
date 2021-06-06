from argparse import ArgumentParser
from MotionGeneratorDataModule import MotionGeneratorDataModule
from MotionGenerator import  MotionGenerator
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_top_k", type=int, default=1)

    parser = MotionGenerator.add_model_specific_args(parser)
    parser = MotionGeneratorDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)


    model = MotionGenerator(**dict_args)
    datamodule = MotionGeneratorDataModule.from_argparse_args(args, tokenizer=model.tokenizer)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=args.save_top_k,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model, datamodule=datamodule)