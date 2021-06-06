from argparse import ArgumentParser
from .model import MotionGeneratorDataModule, MotionGenerator
from transformers import T5TokenizerFast as T5Tokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--save_top_k", type=int, default=1)

    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    datamodule = MotionGeneratorDataModule(batch_size=args.batch_size, tokenizer=tokenizer)
    model = MotionGenerator(args.model_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=args.save_top_k,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        max_epochs=args.max_epochs,
        gpus=args.gpus, 
        progress_bar_refresh_rate=5
    )

    trainer.fit(model, datamodule=datamodule)