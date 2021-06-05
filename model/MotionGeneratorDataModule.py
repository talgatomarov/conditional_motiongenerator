import requests
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from .MotionGeneratorDataset import MotionGeneratorDataset

class MotionGeneratorDataModule(pl.LightningDataModule):
  def __init__(self, batch_size, tokenizer):
    super().__init__()
    self.batch_size = batch_size
    self.tokenizer = tokenizer

  def prepare_data(self):
    response = requests.get('https://docs.google.com/spreadsheets/u/2/d/1qQlqFeJ3iYbzXYrLBMgbmT6LcJLj6JcG3LJyZSbkAJY/export?format=csv')
    assert response.status_code == 200, "Wrong status code"

    os.makedirs("data", exist_ok=True)

    with open('data/motions.csv', 'wb') as f:
      f.write(response.content)

  def setup(self, stage):
    df = pd.read_csv("data/motions.csv")
    df = df[["Infoslide", "Motion"]].dropna(subset=["Infoslide"]).reset_index(drop=True)

    infoslide_encodings, motion_encodings = self.get_encodings(df["Infoslide"], df["Motion"])

    dataset = MotionGeneratorDataset(infoslide_encodings, motion_encodings)

    if stage in (None, 'fit'):
      train_size = int(0.8 * len(dataset))
      val_size = len(dataset) - train_size

      self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

  def get_encodings(self, infoslides, motions):
    infoslide_encodings = self.tokenizer(
      infoslides.tolist(),
      max_length=512,
      padding=True, 
      truncation=True, 
      return_attention_mask=True, 
      add_special_tokens=True, 
      return_tensors="pt", 
      )

    motion_encodings = self.tokenizer(
      motions.tolist(),
      max_length=256,
      padding=True, 
      truncation=True, 
      return_attention_mask=False, 
      add_special_tokens=True, 
      return_tensors='pt'
      )
    
    return infoslide_encodings, motion_encodings

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size)