import torch
from torch.utils.data import Dataset

from yoonimage.data import Dataset1D, Transform1D


class ClassificationDataset(Dataset):
    def __init__(self,
                 dataset: Dataset1D,
                 output_dim: int,
                 transform: Transform1D,
                 ):
        self.data = transform(dataset)
        self.input_dim = self.data.min_channel()
        self.output_dim = output_dim

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        input_ = self.data[item].image.copy_tensor()
        target = self.data[item].label
        return input_, target


def collate_segmentation(tensors):
    # Make the batch order in first step
    inputs = [torch.tensor(data).unsqueeze(dim=0) for data, label in tensors]
    targets = [torch.tensor(label).unsqueeze(dim=0) for data, label in tensors]
    input_tensor = torch.cat(inputs)
    target_tensor = torch.LongTensor(targets)
    return input_tensor, target_tensor
