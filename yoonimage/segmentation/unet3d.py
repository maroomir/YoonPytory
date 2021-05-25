import os.path

import numpy
import torch
import torch.nn
import math
from torch import tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

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


class Convolution3D(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 dRateDropout: float = 0.3):
        super(Convolution3D, self).__init__()
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


class UpSampler3D(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int):
        super(UpSampler3D, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(nDimInput, nDimOutput, kernel_size=2, stride=2, bias=True),
            torch.nn.InstanceNorm3d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, pTensorX: tensor):
        return self.network(pTensorX)


class UNet3D(Module):
    def __init__(self,
                 nDimInput: int = 1,
                 nDimOutput: int = 1,
                 nChannel: int = 8,
                 nCountDepth: int = 4,
                 dRateDropout: float = 0.3):
        super(UNet3D, self).__init__()
        # Init Encoders and Decoders
        self.encoders = torch.nn.ModuleList([Convolution3D(nDimInput, nChannel, dRateDropout)])
        for i in range(nCountDepth - 1):
            self.encoders += [Convolution3D(nChannel, nChannel * 2, dRateDropout)]
            nChannel *= 2
        self.down_sampler = torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        self.worker = Convolution3D(nChannel, nChannel * 2, dRateDropout)
        self.decoders = torch.nn.ModuleList()
        self.up_samplers = torch.nn.ModuleList()
        for i in range(nCountDepth - 1):
            self.up_samplers += [UpSampler3D(nChannel * 2, nChannel)]
            self.decoders += [Convolution3D(nChannel * 2, nChannel, dRateDropout)]
            nChannel //= 2
        self.up_samplers += [UpSampler3D(nChannel * 2, nChannel)]
        self.decoders += [
            torch.nn.Sequential(
                Convolution3D(nChannel * 2, nChannel, dRateDropout),
                torch.nn.Conv3d(nChannel, nDimOutput, kernel_size=1, stride=1),
                torch.nn.Softmax(dim=1)
            )
        ]

    def __padding(self, pTensorX: tensor):
        def floor_ceil(n):
            return math.floor(n), math.ceil(n)

        nBatch, nFlag, nDensity, nHeight, nWidth = pTensorX.shape
        nWidthBitMargin = ((nWidth - 1) | 15) + 1  # 15 = (1111)
        nHeightBitMargin = ((nHeight - 1) | 15) + 1
        nDensityBitMargin = ((nDensity - 1) | 15) + 1
        pPadWidth = floor_ceil((nWidthBitMargin - nWidth) / 2)
        pPadHeight = floor_ceil((nHeightBitMargin - nHeight) / 2)
        pPadDensity = floor_ceil((nDensityBitMargin - nDensity) / 2)
        x = torch.nn.functional.pad(pTensorX, pPadWidth + pPadHeight + pPadDensity)
        return x, (pPadDensity, pPadHeight, pPadWidth, nDensityBitMargin, nHeightBitMargin, nWidthBitMargin)

    def __unpadding(self, x, pPadDensity, pPadHeight, pPadWidth, nDensityMargin, nHeightMargin, nWidthMargin):
        return x[..., pPadDensity[0]:nDensityMargin - pPadDensity[1], pPadHeight[0]:nHeightMargin - pPadHeight[1],
               pPadWidth[0]:nWidthMargin - pPadWidth[1]]

    def forward(self, pTensorX: tensor):
        pTensorX, pPadOption = self.__padding(pTensorX)
        pListStack = []
        pTensorResult = pTensorX
        # Apply down sampling layers
        for i, pEncoder in enumerate(self.encoders):
            pTensorResult = pEncoder(pTensorResult)
            pListStack.append(pTensorResult)
            pTensorResult = self.down_sampler(pTensorResult)
        pTensorResult = self.worker(pTensorResult)
        # Apply up sampling layers
        for pSampler, pDecoder in zip(self.up_samplers, self.decoders):
            pTensorAttached = pListStack.pop()
            pTensorResult = pSampler(pTensorResult)
            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            pPadding = [0, 0, 0, 0]  # left, right, top, bottom
            if pTensorResult.shape[-1] != pTensorAttached.shape[-1]:
                pPadding[1] = 1  # Padding right
            if pTensorResult.shape[-2] != pTensorAttached.shape[-2]:
                pPadding[3] = 1  # Padding bottom
            if sum(pPadding) != 0:
                pTensorResult = torch.nn.functional.pad(pTensorResult, pPadding, "reflect")
            pTensorResult = torch.cat([pTensorResult, pTensorAttached], dim=1)
            pTensorResult = pDecoder(pTensorResult)
        pListStack.clear()  # To Memory Optimizing
        pTensorResult = self.__unpadding(pTensorResult, *pPadOption)
        return pTensorResult


# Define a collate function for the data loader (Assort for Batch)
def collate_tensor(pListTensor):
    pListInput = []
    pListTarget = []
    nHeightMin = min([pData.shape[1] for pData, pTarget in pListTensor]) - 1  # Shape = CH, Y, X
    nWidthMin = min([pData.shape[-1] for pData, pTarget in pListTarget]) - 1  # Shape = CH, Y, X
    for pInputData, pTargetData in pListTensor:
        nStartY = numpy.random.randint(pInputData.shape[1] - nHeightMin)
        nStartX = numpy.random.randint(pInputData.shape[-1] - nWidthMin)
        # Change the tensor type
        pListInput.append(torch.tensor(pInputData[:, nStartY:nStartY + nHeightMin, nStartX:nStartX + nWidthMin]))
        pListTarget.append(torch.tensor(pTargetData[:, nStartY:nStartY + nHeightMin, nStartX:nStartX + nWidthMin]))
    # Grouping batch : Update tensor shape to (Batch, CH, Y, X)
    pListInput = torch.nn.utils.rnn.pad_sequence(pListInput, batch_first=True)
    pListTarget = torch.nn.utils.rnn.pad_sequence(pListTarget, batch_first=True)
    return pListInput, pListTarget


# Define a train function
def __process_train(nEpoch: int, pModel: UNet3D, pDataLoader: DataLoader, pCriterion: BCEWithLogitsLoss,
                    pOptimizer: Adam):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Perform a training using the defined network
    pModel.train()
    # Warp the iterable Data Loader with TQDM
    pBar = tqdm(enumerate(pDataLoader))
    nLengthSample = 0
    nTotalLoss = 0
    nTotalAcc = 0
    for i, (pTensorInput, pTensorTarget) in pBar:
        # Move data and label to device
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.type(torch.FloatTensor).to(pDevice)
        # Pass the input data through the defined network architecture
        pTensorOutput = pModel(pTensorInput)
        # Compute a loss function
        pLoss = pCriterion(pTensorOutput, pTensorTarget)
        nTotalLoss += pLoss.item() * len(pTensorTarget[0])  # Loss per batch * batch
        # Compute network accuracy
        nAcc = torch.sum(torch.eq(pTensorOutput > 0.5, pTensorTarget > 0.5)).item()  # output and targets binary
        nLengthSample += len(pTensorInput[0])
        nTotalAcc += nAcc
        # Perform backpropagation to update network parameters
        pOptimizer.zero_grad()
        pLoss.backward()
        pOptimizer.step()
        pBar.set_description('Epoch:{:3d} [{}/{} {:.2f}%] CE Loss: {:.3f} ACC: {:.2f}%'
                             .format(nEpoch, i, len(pDataLoader), 100.0 * (i / len(pDataLoader)),
                                     nTotalLoss / nLengthSample, (nTotalAcc / nLengthSample) * 100.0))


# Define a test function
def __process_evaluate(pModel: UNet3D, pDataLoader: DataLoader, pCriterion: BCEWithLogitsLoss,
                       pOptimizer: Adam):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Perform an evaluation using the defined network
    pModel.eval()
    # Warp the iterable Data Loader with TQDM
    pBar = tqdm(enumerate(pDataLoader))
    nLengthSample = 0
    nTotalLoss = 0
    nTotalAcc = 0
    for i, (pTensorInput, pTensorTarget) in pBar:
        # Move data and label to device
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorInput.type(torch.FloatTensor).to(pDevice)
        # Pass the input data through the defined network architecture
        pTensorOutput = pModel(pTensorInput)
        # Compute a loss function
        pLoss = pCriterion(pTensorOutput, pTensorTarget)
        nTotalLoss += pLoss.item() * len(pTensorTarget[0])  # Loss per batch * batch
        # Compute network accuracy
        nAcc = torch.sum(torch.eq(pTensorOutput > 0.5, pTensorTarget > 0.5)).item()  # output and targets binary
        nLengthSample += len(pTensorInput[0])
        nTotalAcc += nAcc
    return nTotalLoss / nLengthSample


def train(nEpoch: int, pTrainData: YoonDataset, pValidationData: YoonDataset, strModelPath: str = None,
          bInitEpoch=False):
    dLearningRate = 0.01
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define the training and testing data-set
    pTrainSet = UNetDataset(pTrainData)
    pTrainLoader = DataLoader(pTrainSet, batch_size=4, shuffle=True, collate_fn=collate_tensor,
                              num_workers=8, pin_memory=True)
    pValidationSet = UNetDataset(pValidationData)
    pValidationLoader = DataLoader(pValidationSet, batch_size=1, shuffle=False, collate_fn=collate_tensor,
                                   num_workers=8, pin_memory=True)
    # Define a network model
    pModel = UNet3D().to(pDevice)
    # Set the optimizer with adam
    pOptimizer = torch.optim.Adam(pModel.parameters(), lr=dLearningRate)
    # Set the training criterion
    pCriterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    # Load pre-trained model
    nStart = 0
    print("Directory of the pre-trained model: {}".format(strModelPath))
    if strModelPath is not None and os.path.exists(strModelPath) and bInitEpoch is False:
        pModelData = torch.load(strModelPath)
        nStart = pModelData['epoch']
        pModel.load_state_dict(pModelData['model'])
        pOptimizer.load_state_dict(pModelData['optimizer'])
        print("## Successfully load the model at {} epochs!".format(nStart))
    # Train and Test Repeat
    dMinLoss = 10000.0
    nCountDecrease = 0
    for iEpoch in range(nStart, nEpoch + 1):
        # Train the network
        __process_train(iEpoch, pModel=pModel, pDataLoader=pTrainLoader, pCriterion=pCriterion,
                        pOptimizer=pOptimizer)
        # Test the network
        dLoss = __process_evaluate(pModel=pModel, pDataLoader=pValidationLoader, pCriterion=pCriterion)
        # Save the optimal model
        if dLoss < dMinLoss:
            dMinLoss = dLoss
            torch.save({'epoch': iEpoch, 'model': pModel.state_dict(), 'optimizer': pOptimizer.state_dict()},
                       strModelPath)
            nCountDecrease = 0
        else:
            nCountDecrease += 1
            # Decrease the learning rate by 2 when the test loss decrease 3 times in a row
            if nCountDecrease == 3:
                pDicOptimizerState = pOptimizer.state_dict()
                pDicOptimizerState['param_groups'][0]['lr'] /= 2
                pOptimizer.load_state_dict(pDicOptimizerState)
                print('learning rate is divided by 2')
                nCountDecrease = 0


def test(pTestData: YoonDataset, strModelPath: str = None):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Load UNET model
    pModel = UNet3D().to(pDevice)
    pModel.eval()
    pFile = torch.load(strModelPath)
    pModel.load_state_dict(pFile['model'])
    print("Successfully load the Model in path")
    # Define a data path for plot for test
    pDataSet = UNetDataset(pTestData)
    pDataLoader = DataLoader(pDataSet, batch_size=1, shuffle=False, collate_fn=collate_tensor,
                             num_workers=0, pin_memory=True)
    pBar = tqdm(pDataLoader)
    print("Length of data = ", len(pBar))
    pListOutput = []
    pListTarget = []
    for i, (pTensorInput, pTensorTarget) in enumerate(pBar):
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorOutput = pModel(pTensorInput)
        pListOutput.append(pTensorOutput.detach().cpu().numpy())
        pListTarget.append(pTensorTarget.detach().cpu().numpy())
    # Warp the tensor to Dataset
    pArrayOutput = numpy.concatenate(pListOutput)  # Link the
    return YoonDataset.from_tensor(pArrayOutput=numpy.concatenate(pListOutput))
