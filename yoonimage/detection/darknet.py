from __future__ import division
from typing import Any

import torch
import torch.nn
import torch.nn.functional
from torch import tensor
from torch.nn import Module
from torch.autograd import Function
from torch.utils.data import Dataset

import numpy

from yoonimage.data import YoonDataset


def parse_config(strConfigFile: str):
    pFile = open(strConfigFile, 'r')
    pListLines = pFile.read().split("\n")
    pListLines = [iLine for iLine in pListLines if len(iLine) > 0]  # erase the empty line
    pListLines = [iLine for iLine in pListLines if iLine[0] != "#"]  # erase the comment
    pListLines = [iLine.rstrip().lstrip for iLine in pListLines]  # erase the space
    # Get parameter blocks
    pDicBlock = {}
    pListBlockDict = []
    for iLine in pListLines:
        if iLine[0] == "[":  # start new block
            if iLine(pDicBlock) != 0:
                pListBlockDict.append(pDicBlock)
                pDicBlock = {}
            pDicBlock['type'] = iLine[1:-1].rstrip()
        else:
            strKey, strValue = iLine.split('=')
            pDicBlock[strKey.rstrip()] = strValue.lstrip()
    pListBlockDict.append(pDicBlock)
    return pListBlockDict


class YoloDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset,
                 strConfigFile: str):
        self.data = pDataset
        self.blocks = parse_config(strConfigFile)
        self.width = 0
        self.height = 0
        self.channel = 0

    def __extract_info(self):
        pDicNet = self.blocks[0]
        self.height = int(pDicNet['height'])
        self.width = int(pDicNet['width'])
        self.channel = int(pDicNet['channels'])



class MishActivation(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        ctx.save_for_backward(args[0])
        # Activation :  X * tanh(ln(1 + exp(X)))
        pTensorY = args[0].mul(torch.tanh(torch.nn.functional.softplus(args[0])))
        return pTensorY

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pTensorX = ctx.saved_tensor[0]
        pTensorSigmoid = torch.sigmoid(pTensorX)
        pTensorTanh = torch.nn.functional.softplus(pTensorX).tanh()
        return grad_outputs[0].mul(pTensorTanh + pTensorX * pTensorSigmoid * (1 - pTensorTanh * pTensorTanh))


class Mish(Module):
    def __init__(self,
                 bInplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = bInplace

    def forward(self, pTensorX: tensor):
        return MishActivation.apply(pTensorX)


class MaxPoolStride(Module):
    def __init__(self, nKernel):
        super(MaxPoolStride, self).__init__()
        self.kernel_size = nKernel
        self.pad = nKernel - 1

    def forward(self, pTensorX: tensor):
        pTensorX = torch.nn.functional.pad(pTensorX, (0, self.pad, 0, self.pad), mode="replicate")
        return torch.nn.functional.max_pool2d(pTensorX, self.kernel_size, self.pad)


class DummyLayer(Module):
    def __init__(self):
        super(DummyLayer, self).__init__()


class YoloLayer(Module):
    def __init__(self,
                 pListAnchor: list,
                 nImageWidth: int = 0,
                 nImageHeight: int = 0,
                 nCountClass: int = 0):
        self.image_width = nImageWidth
        self.image_height = nImageHeight
        self.class_count = nCountClass
        self.anchor_boxes = pListAnchor

    def forward(self, pTensorX: tensor):  # Tensor Shape = (Batch, CH, Height, Width)
        nSizeBatch = pTensorX.size(0)
        nCountGrid = self.image_height // pTensorX.size(2)
        nSizeGrid = self.image_height // nCountGrid
        nCountAttribute = 5 + self.class_count  # Count : (x, y, w, h, confidence, class...)
        nCountAnchor = len(self.anchor_boxes)
        pTensorX = pTensorX.view(nSizeBatch, nCountAttribute * nCountAnchor, nSizeGrid * nSizeGrid)
        pTensorX = pTensorX.transpose(1, 2).contiguous()
        pTensorX = pTensorX.view(nSizeBatch, nSizeGrid * nSizeGrid * nCountAnchor, nCountAttribute)
        pListAnchor = [(nWidth / nCountGrid, nHeight / nCountGrid) for nWidth, nHeight in self.anchor_boxes]
        # Sigmoid the CenterX, CenterY and Object Confidence
        pTensorX[:, :, 0] = torch.sigmoid(pTensorX[:, :, 0])  # X
        pTensorX[:, :, 1] = torch.sigmoid(pTensorX[:, :, 1])  # Y
        pTensorX[:, :, 4] = torch.sigmoid(pTensorX[:, :, 4])  # Confidence
        # Add the center offsets
        pArrayGrid = numpy.arance(nSizeGrid)
        pArrayGridX, pArrayGridY = numpy.meshgrid(pArrayGrid, pArrayGrid)
        pDevice = pTensorX.device
        pTensorGridOffsetX = pArrayGridX.view(-1, 1).to(pDevice)
        pTensorGridOffsetY = pArrayGridY.view(-1, 1).to(pDevice)
        pTensorGridOffset = torch.cat((pTensorGridOffsetX, pTensorGridOffsetY), 1) \
            .repeat(1, nCountAnchor).view(-1, 2).unsqueeze(0)
        pTensorX[:, :, :2] += pTensorGridOffset  # X, Y (offset into the item 0, 1)
        # The log area transform to the height and width
        pTensorAnchor = torch.Tensor(pListAnchor)
        pTensorAnchor = pTensorAnchor.repeat(nSizeGrid * nSizeGrid, 1).unsqueeze(0)
        pTensorX[:, :, 2:4] = torch.exp(pTensorX[:, :, 2:4]) * pTensorAnchor  # Width, Height (item 2, 3)
        # Softmax the class scores
        pTensorX[:, :, , 5:5 + self.class_count] = torch.sigmoid((pTensorX[:, :, 5:5 + self.class_count]))
        pTensorX[:, :, :4] *= nCountGrid  # Adjust the stride at the X, Y, Width, Height (item 0 ~ 3)
        return pTensorX


class DarkNet(Module):
    def __init__(self,
                 strConfigFile: str):
        super(DarkNet, self).__init__()
        self.blocks = parse_config(strConfigFile)
        self.net_info, self.modules = self.__create_module()
        self.header = torch.Tensor([0, 0, 0, 0])
        self.seen = 0

    def __create_module(self):
        pDicNet = self.blocks[0]
        pListModule = torch.nn.ModuleList()
        nDimPrev = self.net_info['channels']
        pListFilterStack = []
        for i, iBlock in enumerate(self.blocks[1:]):
            pModule = torch.nn.Sequential()
            if iBlock['type'] == "convolutional":
                strActiveFunc = iBlock['activation']
                try:
                    bBatchNormalize = int(iBlock['batch_normalize'])
                    bBias = False
                except:
                    bBatchNormalize = 0
                    bBias = True
                nDimModule = int(iBlock['filters'])
                bPadding = int(iBlock['pad'])
                nSizeKernel = int(iBlock['size'])
                nStride = int(iBlock['stride'])
                if bPadding > 0:  # Auto padding
                    nPad = (nSizeKernel - 1) // 2
                else:
                    nPad = 0
                # Add the convolutional layer
                pModule.add_module("conv_{0}".format(i),
                                   torch.nn.Conv2d(in_channels=nDimPrev, out_channels=nDimModule,
                                                   kernel_size=nSizeKernel, stride=nStride,
                                                   padding=nPad, bias=bBias))
                # Add the batch norm layer
                if bBatchNormalize > 0:
                    pModule.add_module("batch_norm_{0}".format(i), torch.nn.BatchNorm2d(num_features=nDimModule))
                # Check the activation
                if strActiveFunc == "leaky":
                    pModule.add_module("leaky_{0}".format(i), torch.nn.LeakyReLU(0.1, inplace=True))
                elif strActiveFunc == "mish":
                    pModule.add_module("mish_{0}".format(i), Mish(bInplace=True))
                elif strActiveFunc == "logistic":
                    pModule.add_module("mish_{0}".format(i), torch.nn.Sigmoid())
            elif iBlock['type'] == "upsample":
                nStride = int(iBlock['stride'])
                pModule.add_module("leaky_{0}".format(i), torch.nn.Upsample(scale_factor=nStride, mode="nearest"))
            elif iBlock['type'] == "route":
                iBlock['layers'] = iBlock['layers'].split(',')
                nStart = int(iBlock['layers'][0])
                try:
                    nEnd = int(iBlock['layers'][1])
                except:
                    nEnd = 0
                if nStart > 0:
                    nStart = nStart - i
                if nEnd > 0:
                    nEnd = nEnd - i
                pModule.add_module("route_{0}".format(i), DummyLayer())
                if nEnd < 0:
                    nDimModule = pListFilterStack[i + nStart] + pListFilterStack[i + nEnd]
                else:
                    nDimModule = pListFilterStack[i + nStart]
            # Define the skip connection layer
            elif iBlock['type'] == "shortcut":
                pModule.add_module("shortcut_{0}".format(i), DummyLayer())
            # Define the pooling layer
            elif iBlock['type'] == "maxpool":
                nStride = int(iBlock['stride'])
                nSize = int(iBlock['size'])
                if nStride != 1:
                    pModule.add_module("maxpool_{0}".format(i), torch.nn.MaxPool2d(kernel_size=nSize, stride=nStride))
                else:
                    pModule.add_module("maxpool_{0}".format(i), MaxPoolStride(nKernel=nSize))
            # Define the detection layer
            elif iBlock['type'] == "yolo":
                pListMask = iBlock['mask'].split(",")
                pListMask = [int(strMask) for strMask in pListMask]
                pListAnchorBox = iBlock['anchors'].split(",")
                pListAnchorBox = [int(strAnchor) for strAnchor in pListAnchorBox]
                pListAnchorBox = [(pListAnchorBox[i], pListAnchorBox[i + 1])
                                  for j in range(0, len(pListAnchorBox), 2)]
                pListAnchorBox = [pListAnchorBox[iMask] for iMask in pListMask]
                pModule.add_module("detection_{0}".format(i), YoloLayer(pListAnchorBox))
            pListModule.append(pModule)
            nDimPrev = nDimModule
            pListFilterStack.append(nDimModule)
        return pDicNet, pListModule

    def forward(self, pTensorX: tensor):
        pTensorResult: tensor = None
        pListBlock = self.blocks[1:]
        pDicResultStack = {}
        for i, iBlock in enumerate(pListBlock):
            strType = iBlock['type']
            if strType == "convolutional" or strType == "upsample":
                pTensorX = self.modules[i](pTensorX)
                pDicResultStack[i] = pTensorX
            elif strType == "route":
                pListLayer = iBlock['layers']
                pListLayer = [int(j) for j in pListLayer]
                if pListLayer[0] > 0:
                    pListLayer[0] = pListLayer[0] - i
                if len(pListLayer) == 1:
                    pTensorX = pDicResultStack[i + pListLayer[0]]
                else:
                    if (pListLayer[1]) > 0:
                        pListLayer[1] = pListLayer[1] - i
                    pTensorMap1 = pDicResultStack[i + pListLayer[0]]
                    pTensorMap2 = pDicResultStack[i + pListLayer[1]]
                    pTensorX = torch.cat((pTensorMap1, pTensorMap2), 1)
                    pDicResultStack[i] = pTensorX
            elif strType == "shortcut":
                nFromBack = int(iBlock['from'])
                pTensorX = pDicResultStack[i - 1] + pDicResultStack[i + nFromBack]
                pDicResultStack[i] = pTensorX
            elif strType == "yolo":
                self.modules[i][0].image_height = int(self.net_info['height'])  # YoloLayer
                self.modules[i][0].image_width = int(self.net_info['width'])
                self.modules[i][0].class_count = int(iBlock['classes'])
                # Predict the bounding boxes
                pTensorX = pTensorX.data
                pTensorX = self.modules[i][0](pTensorX)
                if type(pTensorX) == int:
                    continue
                if pTensorResult is None:
                    pTensorResult = pTensorX
                else:
                    pTensorResult = torch.cat((pTensorResult, pTensorX), dim=1)
                pDicResultStack[i] = pDicResultStack[i - 1]
        return pTensorResult if pTensorResult is not None else 0

    def load_weight(self, strWeightFilePath: str):
        # Open the weights file
        pFile = open(strWeightFilePath, "rb")
        # The first values are a header information
        # Major version / Minor version / Subversion / Images
        pHeader = numpy.fromfile(pFile, dtype=numpy.int32, count=5)
        self.header = torch.from_numpy(pHeader)
        self.seen = self.header[3]
        # The rest of the values are the weights
        pArrayReadWeight = numpy.fromfile(pFile, dtype=numpy.float32)
        pListBlock = self.blocks[1:]
        nStart = 0
        for i, iBlock in enumerate(pListBlock):
            strType = iBlock['type']
            if strType == "convolutional":
                try:
                    bBatchNormalize = int(iBlock['batch_normalize'])
                except:
                    bBatchNormalize = 0
                pConvolution = self.modules[i][0]
                if bBatchNormalize > 0:
                    pNormalized = self.modules[i][1]
                    # Get the number of weights of Batch Norm Layer
                    nCountBias = pNormalized.bias.numel()
                    # Load the weights
                    pArrayBiases = torch.from_numpy(pArrayReadWeight[nStart: nStart + nCountBias])
                    nStart += nCountBias
                    pArrayWeight = torch.from_numpy(pArrayReadWeight[nStart: nStart + nCountBias])
                    pArrayRunningMean = torch.from_numpy(pArrayReadWeight[nStart: nStart + nCountBias])
                    nStart += nCountBias
                    pArrayRunningVar = torch.from_numpy(pArrayReadWeight[nStart: nStart + nCountBias])
                    nStart += nCountBias
                    # Cast the loaded weights into the dimension of the model weight
                    pArrayBiases = pArrayBiases.view_as(pNormalized.bias.data)
                    pArrayWeight = pArrayWeight.view_as(pNormalized.weight.data)
                    pArrayRunningMean = pArrayRunningMean.view_as(pNormalized.running_mean)
                    pArrayRunningVar = pArrayRunningVar.view_as(pNormalized.running_var)
                    # Copy the loaded weights to model
                    pNormalized.bias.data.copy_(pArrayBiases)
                    pNormalized.weight.data.copy_(pArrayWeight)
                    pNormalized.running_mean.copy_(pArrayRunningMean)
                    pNormalized.running_var.copy_(pArrayRunningVar)
                else:
                    # Get the number of biases
                    nCountBias = pConvolution.bias.numel()
                    # Load the weights
                    pArrayBiases = torch.from_numpy(pArrayReadWeight[nStart: nStart + nCountBias])
                    nStart += nCountBias
                    # Reshape the loaded weights according to the dimension of the model weight
                    pArrayBiases = pArrayBiases.view_as(pConvolution.bias.data)
                    # Copy the loaded weights to model
                    pConvolution.bias.data.copy_(pArrayBiases)
                # Load the weights for convolutional layer
                nCountWeight = pConvolution.weight.numel()
                pArrayWeight = torch.from_numpy(pArrayReadWeight[nStart: nStart + nCountWeight])
                nStart += nCountWeight
                # Cast the loaded weights into the dimension of the model weight
                pArrayWeight = pArrayWeight.view_as(pConvolution.weight.data)
                # Copy the loaded weights to model
                pConvolution.weight.data.copy_(pArrayWeight)

    def save_weight(self, strWeightFilePath: str):
        def to_cpu(pTensor: tensor):
            if pTensor.is_cuda:
                return torch.FloatTensor(pTensor.size()).copy_(pTensor)
            else:
                return pTensor

        pFile = open(strWeightFilePath, "wb")
        # Attach the header at the top of the file
        self.header[3] = self.seen
        self.header.numpy().tofile(pFile)
        # Save the weight to file directly
        pListBlock = self.blocks[1:]
        for i, iBlock in enumerate(pListBlock):
            strType = iBlock['type']
            if strType == "convolutional":
                try:
                    bBatchNormalize = int(iBlock['batch_normalize'])
                except:
                    bBatchNormalize = 0
                pConvolution = self.modules[i][0]
                if bBatchNormalize > 0:
                    pNormalized = self.modules[i][1]
                    to_cpu(pNormalized.bias.data).numpy().tofile(pFile)
                    to_cpu(pNormalized.weight.data).numpy().tofile(pFile)
                    to_cpu(pNormalized.running_mean).numpy().tofile(pFile)
                    to_cpu(pNormalized.running_var).numpy().tofile(pFile)
                else:
                    to_cpu(pConvolution.bias.data).numpy().tofile(pFile)
                to_cpu(pConvolution.weight.data).numpy().tofile(pFile)

