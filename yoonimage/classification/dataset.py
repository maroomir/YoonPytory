import torch
from torch.utils.data import Dataset

from yoonimage.data import YoonDataset


class Decimalize(object):
    def __call__(self, pDataset: YoonDataset):
        return pDataset.decimalize()


class Recover(object):
    def __call__(self, pDataset: YoonDataset):
        return pDataset.recover()


class Normalization(object):
    def __init__(self,
                 pNormalizeMean: list = None,
                 pNormalizeStd: list = None):
        self.means = pNormalizeMean if isinstance(pNormalizeMean, list) else [pNormalizeMean]
        self.stds = pNormalizeStd if isinstance(pNormalizeStd, list) else [pNormalizeStd]
        assert len(self.means) == len(self.stds)
        assert 0 < len(self.means) <= 3

    def __call__(self, pDataset: YoonDataset):
        for iChannel in range(len(self.means)):
            pDataset.normalize(iChannel, dMean=self.means[iChannel], dStd=self.stds[iChannel])
        return pDataset


class Denormalization(object):
    def __init__(self,
                 pNormalizeMean: list = None,
                 pNormalizeStd: list = None):
        self.means = pNormalizeMean if isinstance(pNormalizeMean, list) else [pNormalizeMean]
        self.stds = pNormalizeStd if isinstance(pNormalizeStd, list) else [pNormalizeStd]
        assert len(self.means) == len(self.stds)
        assert 0 < len(self.means) <= 3

    def __call__(self, pDataset: YoonDataset):
        for iChannel in range(len(self.means)):
            pDataset.denormalize(iChannel, dMean=self.means[iChannel], dStd=self.stds[iChannel])
        return pDataset


class ZNormalization(object):
    def __call__(self, pDataset: YoonDataset):
        return pDataset.normalize(strOption="z")


class MinMaxNormalization(object):
    def __call__(self, pDataset: YoonDataset):
        return pDataset.normalize(strOption="minmax")


class Resize(object):
    def __init__(self,
                 nWidth=0,
                 nHeight=0):
        self.width = 0
        self.height = 0
        self.option = "min" if nWidth == 0 and nHeight == 0 else None

    def __call__(self, pDataset: YoonDataset):
        return pDataset.resize(nWidth=self.width, nHeight=self.height, strOption=self.option)


class Rechannel(object):
    def __init__(self,
                 nChannel=0):
        self.channel = nChannel
        self.option = "min" if nChannel == 0 else None

    def __call__(self, pDataset: YoonDataset):
        return pDataset.rechannel(nChannel=self.channel, strOption=self.option)


class ClassificationDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset,
                 nDimOutput: int,
                 *args,  # Resize, Rechannel...
                 ):

        self.data = pDataset
        self.input_dim = self.data.min_channel()
        self.output_dim = nDimOutput
        if len(args) > 0:
            for pTransform in args:
                pTransform(self.data)

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
