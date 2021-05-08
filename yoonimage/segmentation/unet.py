import numpy
import torch
import torch.nn
from torch.nn import Module
from torch.utils.data import Dataset
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
        # Rescaling 0-255 to 0-1
        pArrayTarget = self.targets[item].image.normalization(dMean=0, dStd=255.0).copy_tensor()
        pArrayInput = self.inputs[item].image.normalization(dMean=0, dStd=255.0).copy_tensor()
        return pArrayInput, pArrayTarget


class UNet(Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Down-sampling
        self.encoder1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=64),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=64),
                                            torch.nn.ReLU()
                                            )
        self.down_sampler1 = torch.nn.MaxPool2d(kernel_size=2)
        self.encoder2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=128),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=128),
                                            torch.nn.ReLU()
                                            )
        self.down_sampler2 = torch.nn.MaxPool2d(kernel_size=2)
        self.encoder3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=256),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=256),
                                            torch.nn.ReLU()
                                            )
        self.down_sampler3 = torch.nn.MaxPool2d(kernel_size=2)
        self.encoder4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=512),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=512),
                                            torch.nn.ReLU()
                                            )
        self.down_sampler4 = torch.nn.MaxPool2d(kernel_size=2)
        self.encoder5 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=1024),
                                            torch.nn.ReLU()
                                            )
        # Up-sampling
        self.decoder5 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=512),
                                            torch.nn.ReLU()
                                            )
        self.up_sampler4 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2,
                                                    stride=2, padding=0, bias=True)
        self.decoder4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=512 * 2, out_channels=512, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=512),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=256),
                                            torch.nn.ReLU()
                                            )
        self.up_sampler3 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2,
                                                    stride=2, padding=0, bias=True)
        self.decoder3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=256),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=128),
                                            torch.nn.ReLU()
                                            )
        self.up_sampler2 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2,
                                                    stride=2, padding=0, bias=True)
        self.decoder2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=128),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=64),
                                            torch.nn.ReLU()
                                            )
        self.up_sampler1 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2,
                                                    stride=2, padding=0, bias=True)
        self.decoder1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=64),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                            torch.nn.BatchNorm2d(num_features=64),
                                            torch.nn.ReLU()
                                            )
        self.fc_layer = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2,
                                                 stride=2, padding=0, bias=True)

    def forward(self, pTensorX):
        # Down-sampling
        pEncoder1 = self.encoder1(pTensorX)
        pEncoder2 = self.encoder2(self.down_sampler1(pEncoder1))
        pEncoder3 = self.encoder3(self.down_sampler2(pEncoder2))
        pEncoder4 = self.encoder4(self.down_sampler3(pEncoder3))
        pEncoder5 = self.encoder5(self.down_sampler4(pEncoder4))
        # Up-sampling
        pDecoder5 = self.decoder5(pEncoder5)
        pTensorChain4 = torch.cat((self.up_sampler4(pDecoder5), pEncoder4), dim=1)  # 1 : To-Channel
        pDecoder4 = self.decoder4(pTensorChain4)
        pTensorChain3 = torch.cat((self.up_sampler3(pDecoder4), pEncoder3), dim=1)  # 1 : To-Channel
        pDecoder3 = self.decoder3(pTensorChain3)
        pTensorChain2 = torch.cat((self.up_sampler2(pDecoder3), pEncoder2), dim=1)  # 1 : To-Channel
        pDecoder2 = self.decoder2(pTensorChain2)
        pTensorChain1 = torch.cat((self.up_sampler1(pDecoder2), pEncoder1), dim=1)  # 1 : To-Channel
        pDecoder1 = self.decoder1(pTensorChain1)
        # Full-convolution Layer
        pTensorResult = self.fc_layer(pDecoder1)
        return pTensorResult
