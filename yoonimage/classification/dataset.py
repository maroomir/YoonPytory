import torch
from torch.utils.data import Dataset

from yoonimage.data import YoonDataset


class ClassificationDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset,
                 nDimOutput: int,
                 *args  # "resize", "rechannel", "z_norm", "minmax_norm"
                 ):
        def transform():
            if "resize" in args:
                self.data.resize(strOption="min")
            if "rechannel" in args:
                self.data.rechannel(strOption="min")
            if "z_norm" in args:
                self.data.normalize(strOption="z")
            if "minmax_norm" in args:
                self.data.normalize(strOption="minmax")

        self.data = pDataset
        if len(args) > 0:
            transform()
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