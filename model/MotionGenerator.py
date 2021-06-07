from argparse import ArgumentParser
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW

class MotionGenerator(pl.LightningModule):
  def __init__(self, learning_rate, model_name, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name, return_dict=True)
    self.tokenizer= T5Tokenizer.from_pretrained(self.hparams.model_name)

  def forward(self, input_ids,  attention_mask, labels=None):
    output = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True) 

    return loss

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True) 

    return loss

  def configure_optimizers(self):
    return AdamW(self.parameters(), lr=self.hparams.learning_rate)

  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("MotionGenerator")
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--model_name', type=str, default="t5-small")
    return parent_parser