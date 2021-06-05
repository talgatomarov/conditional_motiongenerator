from .model import MotionGeneratorDataModule, MotionGenerator
from transformers import T5TokenizerFast as T5Tokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    datamodule = MotionGeneratorDataModule(batch_size=32, tokenizer=tokenizer)
    model = MotionGenerator()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        max_epochs=50,
        gpus=1, 
        progress_bar_refresh_rate=5
    )

    trainer.fit(model, datamodule=datamodule)

    model.save_pretrained("conditional_motion_generator")

