import torch
from torch.utils.data import Dataset
from yoonspeech.parser import YoonParser


# Define a SpeakerDataset class
class SpeakerDataset(Dataset):
    def __init__(self,
                 pParser: YoonParser,
                 strType: str = "train"):
        self.parser = pParser
        self.type = strType

    def __len__(self):
        if self.type == "train":
            return self.parser.get_train_count()
        elif self.type == "test":
            return self.parser.get_test_count()
        else:
            Exception("Speaker Dataset type is mismatching")

    def __getitem__(self, item):  # obtain label and file name
        if self.type == "train":
            nLabel = self.parser.get_train_label(item)
            pArrayFeature = self.parser.get_train_data(item)
            return nLabel, pArrayFeature
        elif self.type == "test":
            nLabel = self.parser.get_test_label(item)
            pArrayFeature = self.parser.get_test_data(item)
            return nLabel, pArrayFeature
        else:
            Exception("Speaker Dataset type is mismatching")
