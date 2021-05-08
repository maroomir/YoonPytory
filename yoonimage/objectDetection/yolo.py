from __future__ import division
import yoonimage.objectDetection
import time
import torch
import torch.nn
from torch.autograd import Variable
import numpy
import cv2
import argparse
import os
import os.path
import darknet
import pickle
import pandas
import random

class YoloLayer(torch.nn.Module):
    def __init__(self,
                 pListAnchor: list,
                 nCountClass: int,
                 pPairDimInput = yoonimage.objectDetection.YOLO_SIZE_NORMAL):
        super(YoloLayer, self).__init__()
        self.anchors = pListAnchor
        self.anchorCount = len(pListAnchor)
        self.classCount = len(nCountClass)
        self.threshold = 0.5
        self.objectScale = 1
        self.otherScale = 100
        self.metrics = {}
        self.dimension = pPairDimInput[0]
        self.gridSize = 0
        self.stride = 0

    def compute_grid_offset(self,
                            nSize: int):
        if torch.cuda.is_available():
            pDevice = torch.device('cuda')
        else:
            pDevice = torch.device('cpu')
        self.gridSize = nSize
        self.stride = self.dimension / self.gridSize
        # Calculate offsets for each grid
        self.


