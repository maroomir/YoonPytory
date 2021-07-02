import torch
from torch.utils.data import Dataset

from yoonimage.data import YoonDataset, YoonTransform


class ClassificationDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset,
                 nDimOutput: int,
                 pTransform: YoonTransform,
                 ):
        self.data = pTransform(pDataset)
        self.input_dim = self.data.min_channel()
        self.output_dim = nDimOutput

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        pArrayInput = self.data[item].image.copy_tensor()
        nTarget = self.data[item].label
        return pArrayInput, nTarget


def collate_segmentation(pListTensor):
    # Make the batch order in first step
    pListInput = [torch.tensor(pData).unsqueeze(dim=0) for pData, nLabel in pListTensor]
    pListTarget = [torch.tensor(nLabel).unsqueeze(dim=0) for pData, nLabel in pListTensor]
    pTensorInput = torch.cat(pListInput, dim=0)
    pTensorTarget = torch.LongTensor(pListTarget)
    return pTensorInput, pTensorTarget
