from argparse import ArgumentParser
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer, AdamW

class MotionGenerator(pl.LightningModule):
  def __init__(self, model_name="t5-small", learning_rate=0.0001):
    super().__init__()
    self.save_hyperparameters()
    self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

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

  def get_tokenizer(self):
    return T5Tokenizer.from_pretrained(self.hparams.model_name)

  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--model_name', type=str, default="t5-small")
    return parser