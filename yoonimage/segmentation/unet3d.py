import os.path

import numpy
import torch
import torch.nn
import torch.nn.functional
from tqdm import tqdm
from torch import tensor
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from yoonimage.data import YoonDataset


class UNetDataset(Dataset):
    def __init__(self,
                 pInput: YoonDataset,
                 pTarget: YoonDataset):
        self.targets = pTarget
        self.inputs = pInput

    def __len__(self):
        return self.targets.__len__()

    def __getitem__(self, item):
        pArrayTarget = self.targets[item].image.copy_tensor()
        pArrayInput = self.inputs[item].image.copy_tensor()
        return pArrayInput, pArrayTarget


class Parallelize(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 dRateDropout: float = 0.3):
        super(Parallelize, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv3d(nDimInput, nDimOutput, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm3d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout3d(dRateDropout),
            torch.nn.Conv3d(nDimOutput, nDimOutput, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm3d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout3d(dRateDropout)
        )

    def forward(self, pTensorX: tensor):
        return self.network(pTensorX)


class UpSampler(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int):
        super(UpSampler, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(nDimInput, nDimOutput, kernel_size=2, stride=2, bias=True),
            torch.nn.InstanceNorm3d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, pTensorX: tensor):
        return self.network(pTensorX)


class UNet3D(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 nChannel: int,
                 nCountDepth: int,
                 dRateDropout: float = 0.3):
        super(UNet3D, self).__init__()
        self.encoders = torch.nn.ModuleList([Parallelize(nDimInput, nChannel, dRateDropout)])
        for i in range(nCountDepth - 1):
            self.encoders += [Parallelize(nChannel, nChannel * 2, dRateDropout)]
            nChannel *= 2
        self.down_sampler = torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        self.worker = Parallelize(nChannel, nChannel * 2, dRateDropout)
        self.decoders = torch.nn.ModuleList()
        self.up_samplers = torch.nn.ModuleList()
        for i in range(nCountDepth - 1):
            self.up_samplers += [UpSampler(nChannel * 2, nChannel)]
            self.decoders += [Parallelize(nChannel * 2, nChannel, dRateDropout)]
            nChannel //= 2
        self.up_samplers += [UpSampler(nChannel * 2, nChannel)]
        self.decoders += [
            torch.nn.Sequential(
                Parallelize(nChannel * 2, nChannel, dRateDropout),
                torch.nn.Conv3d(nChannel, nDimOutput, kernel_size=1, stride=1),
                torch.nn.Softmax(dim=1)
            )
        ]

    def forward(self, pTensorX: tensor):
        pass