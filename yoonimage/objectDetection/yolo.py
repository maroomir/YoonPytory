from __future__ import division

import torch
import torch.nn
import torch.nn.functional
from torch import tensor
from torch.nn import Module
from torch.autograd import Variable
from yoonpytory.vector import YoonVector2D
from yoonimage.image import YoonImage
import numpy


class DummyLayer(Module):
    def __init__(self):
        super(DummyLayer, self).__init__()


class DetectionLayer(Module):
    def __init__(self, pListAnchor):
        super(DetectionLayer, self).__init__()
        self.anchors = pListAnchor


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


def create_module(pListBlockDict: list):
    pDicNet = pListBlockDict[0]
    pListModule = torch.nn.ModuleList()
    nCountInput = 3
    pListFilterStack = []
    for i, iBlock in enumerate(pListBlockDict[1:]):
        pModule = torch.nn.Sequential()

        if iBlock['type'] == "convolutional":
            strActiveFunc = iBlock['activation']
            try:
                bBatchNormalize = int(iBlock['batch_normalize'])
                bBias = False
            except:
                bBatchNormalize = 0
                bBias = True
            nCountFilter = int(iBlock['filters'])
            bPadding = int(iBlock['pad'])
            nSizeKernel = int(iBlock['size'])
            nStride = int(iBlock['stride'])
            if bPadding > 0:
                nPad = (nSizeKernel - 1) // 2
            else:
                nPad = 0
            # Add the convolutional layer
            pModule.add_module("conv_{0}".format(i), torch.nn.Conv2d(in_channels=nCountInput, out_channels=nCountFilter,
                                                                     kernel_size=nSizeKernel, stride=nStride,
                                                                     padding=nPad, bias=bBias))
            # Add the batch norm layer
            if bBatchNormalize > 0:
                pModule.add_module("batch_norm_{0}".format(i), torch.nn.BatchNorm2d(num_features=nCountFilter))
            # Check the activation
            if strActiveFunc == "leaky":
                pModule.add_module("leaky_{0}".format(i), torch.nn.LeakyReLU(0.1, inplace=True))

        elif iBlock['type'] == "upsample":
            nStride = int(iBlock['stride'])
            pModule.add_module("leaky_{0}".format(i), torch.nn.Upsample(scale_factor=2, mode="nearest"))

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
                nCountFilter = pListFilterStack[i + nStart] + pListFilterStack[i + nEnd]
            else:
                nCountFilter = pListFilterStack[i + nStart]
        # Define the skip connection layer
        elif iBlock['type'] == "shortcut":
            pModule.add_module("shortcut_{0}".format(i), DummyLayer())
        # Define the detection layer
        elif iBlock['type'] == "yolo":
            pListMask = iBlock['mask'].split(",")
            pListMask = [int(strMask) for strMask in pListMask]
            pListAnchor = iBlock['anchors'].split(",")
            pListAnchor = [int(strAnchor) for strAnchor in pListAnchor]
            pListAnchor = [YoonVector2D(pListAnchor[j], pListAnchor[j + 1]) for j in range(0, len(pListAnchor), 2)]
            pListAnchor = [pListAnchor[iMask] for iMask in pListMask]
            pModule.add_module("detection_{0}".format(i), DetectionLayer(pListAnchor))
        pListModule.append(pModule)
        pListFilterStack.append(nCountFilter)
    return pDicNet, pListModule

class Dartnet(Module):
    def __init__(self, strConfigFile):
        super(Dartnet, self).__init__()
        self.blocks = parse_config(strConfigFile)
        self.imageHeight: int = 0
        self.classCount: int = 0
        self.net_info, self.modules = create_module(self, self.blocks)

    def forward(self, pTensorX: tensor):
        pListBlock = self.blocks[1:]
        pListTensorStack = []
        bWrite = False
        for i, iBlock in enumerate(pListBlock):
            strType = iBlock['type']
            if strType == "convolutional" or strType == "upsample":
                pTensorResult = self.modules[i](pTensorX)
            elif strType == "route":
                pListLayer = iBlock['layers']
                pListLayer = [int(j) for j in pListLayer]
                if pListLayer[0] > 0:
                    pListLayer[0] = pListLayer[0] - i
                if len(pListLayer) == 1:
                    pTensorResult = pListTensorStack[i + pListLayer[0]]
                else:
                    if (pListLayer[1]) > 0:
                        pListLayer[1] = pListLayer[1] - i
                    pTensorMap1 = pListTensorStack[i + pListLayer[0]]
                    pTensorMap2 = pListTensorStack[i + pListLayer[1]]
                    pTensorResult = torch.cat((pTensorMap1, pTensorMap2), 1)
            elif strType == "shortcut":
                nFromBack = int(iBlock['from'])
                pTensorResult = pListTensorStack[i-1] + pListTensorStack[i+nFromBack]
            elif strType == "yolo":
                pListAnchor = self.modules[i][0].anchors
                self.imageHeight = int(self.net_info['height'])
                self.classCount = int(iBlock['classes'])
                # Predict the bounding boxes


    def predict_boxes(self, pTensorOutput: tensor):  # Tensor Shape = (Batch, CH, Height, Width)
        nSizeBatch = pTensorOutput.size(0)
        nCountGrid = self.imageHeight // pTensorOutput.size(2)
        nSizeGrid = self.imageHeight // nCountGrid
        pTensorOutput = pTensorOutput.size(1)