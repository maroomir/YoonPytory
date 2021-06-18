import torch
import torch.nn.functional
from torch.nn import Module
from torch import tensor


class ConvolutionBlock(Module):
    def __init__(self,
                 nDimInput: int,
                 pListFilter: list,
                 nStride: int):
        super(ConvolutionBlock, self).__init__()
        self.filter1, self.filter2, self.filter3 = pListFilter
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(nDimInput, self.filter1, kernel_size=1, stride=nStride, padding=0, bias=False),
            torch.nn.BatchNorm2d(self.filter1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.filter1, self.filter2, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.filter2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.filter2, self.filter3, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(self.filter3)
        )
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(nDimInput, self.filter3, kernel_size=1, stride=nStride, bias=False),
            torch.nn.BatchNorm2d(self.filter3)
        )

    def forward(self, pTensorX: tensor):
        pTensorOut = self.network(pTensorX)
        pTensorOut += self.shortcut(pTensorX)
        pTensorOut = torch.nn.functional.relu(pTensorOut)
        return pTensorOut


class IdentityBlock(Module):
    def __init__(self,
                 pListFilter: list):
        self.filter1, self.filter2, self.filter3 = pListFilter
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(self.filter3, self.filter1, kernel_size=1, stride=1, padding=0, bias=False),  # Pad=0=valid
            torch.nn.BatchNorm2d(self.filter1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.filter1, self.filter2, kernel_size=3, stride=1, padding=1, bias=False),  # Pad=1=same
            torch.nn.BatchNorm2d(self.filter2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.filter2, self.filter3, kernel_size=1, stride=1, padding=0, bias=False),  # Pad=0=valid
            torch.nn.BatchNorm2d(self.filter3)
        )

    def forward(self, pTensorX: tensor):
        pTensorOut = self.network(pTensorX),
        pTensorOut += pTensorX
        pTensorOut = torch.nn.functional.relu(pTensorOut)
        return pTensorOut


class ResNet50(Module):  # Conv Count = 50
    def __init__(self,
                 nCountClass=10):
        super(ResNet50, self).__init__()
        self.layer1 = torch.nn.Sequential(  # Conv=1
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.layer2 = torch.nn.Sequential(  # Conv=10
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvolutionBlock(nDimInput=64, pListFilter=[64, 64, 256], nStride=1),  # Conv=4
            IdentityBlock(pListFilter=[64, 64, 256]),   # Conv=3
            IdentityBlock(pListFilter=[64, 64, 256])    # Conv=3
        )
        self.layer3 = torch.nn.Sequential(  # Conv=13
            ConvolutionBlock(nDimInput=256, pListFilter=[128, 128, 512], nStride=2),
            IdentityBlock(pListFilter=[128, 128, 512]),
            IdentityBlock(pListFilter=[128, 128, 512]),
            IdentityBlock(pListFilter=[128, 128, 512])
        )
        self.layer4 = torch.nn.Sequential(  # Conv=16
            ConvolutionBlock(nDimInput=512, pListFilter=[256, 256, 1024], nStride=2),
            IdentityBlock(pListFilter=[256, 256, 1024]),
            IdentityBlock(pListFilter=[256, 256, 1024]),
            IdentityBlock(pListFilter=[256, 256, 1024]),
            IdentityBlock(pListFilter=[256, 256, 1024])
        )
        self.layer5 = torch.nn.Sequential(  # Conv=10
            ConvolutionBlock(nDimInput=1024, pListFilter=[512, 512, 2048], nStride=2),
            IdentityBlock(pListFilter=[512, 512, 2048]),
            IdentityBlock(pListFilter=[512, 512, 2048])
        )
        self.fc_layer = torch.nn.Linear(2048, nCountClass)

    def forward(self, pTensorX: tensor):
        pTensorOut = self.layer1(pTensorX)
        pTensorOut = self.layer2(pTensorOut)
        pTensorOut = self.layer3(pTensorOut)
        pTensorOut = self.layer4(pTensorOut)
        pTensorOut = self.layer5(pTensorOut)
        pTensorOut = torch.nn.functional.avg_pool2d(pTensorOut, kernel_size=1)
        pTensorOut = pTensorOut.view(pTensorOut.size(0), -1)
        pTensorOut = self.fc_layer(pTensorOut)
        return pTensorOut