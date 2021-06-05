from torch.utils.data import Dataset

class MotionGeneratorDataset(Dataset):
    def __init__(self, infoslide_encodings, motion_encodings):
      self.infoslide_encodings = infoslide_encodings
      self.motion_encodings = motion_encodings

    def __getitem__(self, idx):
      return {'input_ids': self.infoslide_encodings["input_ids"][idx],
              'labels': self.motion_encodings["input_ids"][idx],
              'attention_mask': self.infoslide_encodings["attention_mask"][idx]}

    def __len__(self):
      return len(self.infoslide_encodings.input_ids)