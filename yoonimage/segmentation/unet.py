import math
import os.path

import numpy
import torch
import torch.nn
import torch.nn.functional
from torch import tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from yoonimage.data import YoonDataset
from yoonpytory.log import YoonNLM


class UNetDataset(Dataset):
    def __init__(self,
                 pInput: YoonDataset,
                 pTarget: YoonDataset = None,
                 nDimOutput=1
                 ):
        self.inputs = pInput
        self.inputs.resize(strOption="min")
        self.inputs.rechannel(strOption="min")
        self.inputs.normalize(strOption="z")
        self.input_dim = self.inputs.min_channel()
        self.targets = pTarget
        self.output_dim = nDimOutput

    def __len__(self):
        return self.inputs.__len__()

    def __getitem__(self, item):
        if self.targets is None:
            pArrayInput = self.inputs[item].image.copy_tensor()
            return pArrayInput
        else:
            pArrayTarget = self.targets[item].image.copy_tensor()
            pArrayInput = self.inputs[item].image.copy_tensor()
            return pArrayInput, pArrayTarget


class ConvolutionBlock(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 dRateDropout: float = 0.3):
        super(ConvolutionBlock, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(nDimInput, nDimOutput, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm2d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout2d(dRateDropout),
            torch.nn.Conv2d(nDimOutput, nDimOutput, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm2d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout2d(dRateDropout)
        )

    def forward(self, pTensorX: tensor):
        return self.network(pTensorX)


class UpSamplerBlock(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int):
        super(UpSamplerBlock, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nDimInput, nDimOutput, kernel_size=2, stride=2, bias=True),
            torch.nn.InstanceNorm2d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, pTensorX: tensor):
        return self.network(pTensorX)


class UNet2D(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 nChannel: int,
                 nCountDepth: int,
                 dRateDropout: float = 0.3):
        super(UNet2D, self).__init__()
        # Init Encoders and Decoders
        self.encoders = torch.nn.ModuleList([ConvolutionBlock(nDimInput, nChannel, dRateDropout)])
        for i in range(nCountDepth - 1):
            self.encoders += [ConvolutionBlock(nChannel, nChannel * 2, dRateDropout)]
            nChannel *= 2
        self.down_sampler = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.worker = ConvolutionBlock(nChannel, nChannel * 2, dRateDropout)
        self.decoders = torch.nn.ModuleList()
        self.up_samplers = torch.nn.ModuleList()
        for i in range(nCountDepth - 1):
            self.up_samplers += [UpSamplerBlock(nChannel * 2, nChannel)]
            self.decoders += [ConvolutionBlock(nChannel * 2, nChannel, dRateDropout)]
            nChannel //= 2
        self.up_samplers += [UpSamplerBlock(nChannel * 2, nChannel)]
        self.decoders += [
            torch.nn.Sequential(
                ConvolutionBlock(nChannel * 2, nChannel, dRateDropout),
                torch.nn.Conv2d(nChannel, nDimOutput, kernel_size=1, stride=1),
                torch.nn.Tanh()
            )
        ]

    def __padding(self, pTensorX: tensor):
        def floor_ceil(n):
            return math.floor(n), math.ceil(n)

        nBatch, nDensity, nHeight, nWidth = pTensorX.shape
        nWidthBitMargin = ((nWidth - 1) | 15) + 1  # 15 = (1111)
        nHeightBitMargin = ((nHeight - 1) | 15) + 1
        pPadWidth = floor_ceil((nWidthBitMargin - nWidth) / 2)
        pPadHeight = floor_ceil((nHeightBitMargin - nHeight) / 2)
        x = torch.nn.functional.pad(pTensorX, pPadWidth + pPadHeight)
        return x, (pPadHeight, pPadWidth, nHeightBitMargin, nWidthBitMargin)

    def __unpadding(self, x, pPadHeight, pPadWidth, nHeightMargin, nWidthMargin):
        return x[...,
               pPadHeight[0]:nHeightMargin - pPadHeight[1],
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


# Define a train function
def __process_train(pModel: UNet2D, pDataLoader: DataLoader, pCriterion: BCEWithLogitsLoss,
                    pOptimizer: Adam, pLog: YoonNLM):
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
    dTotalLoss = 0.0
    dTotalAcc = 0.0
    for i, (pTensorInput, pTensorTarget) in pBar:
        # Move data and label to device
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.type(torch.FloatTensor).to(pDevice)
        # Pass the input data through the defined network architecture
        pTensorOutput = pModel(pTensorInput)
        # Compute a loss function
        pLoss = pCriterion(pTensorOutput, pTensorTarget)
        dTotalLoss += pLoss.item() * len(pTensorTarget[0])  # Loss per batch * batch
        # Compute network accuracy
        nAcc = torch.sum(torch.eq(pTensorOutput > 0.5, pTensorTarget > 0.5)).item()  # output and targets binary
        nLengthSample += len(pTensorInput[0])
        dTotalAcc += nAcc
        # Perform backpropagation to update network parameters
        pOptimizer.zero_grad()
        pLoss.backward()
        pOptimizer.step()
        strMessage = pLog.write(i, len(pDataLoader),
                                CELoss=dTotalLoss / nLengthSample, ACC=(dTotalAcc / nLengthSample) * 100.0)
        pBar.set_description(strMessage)


# Define a test function
def __process_evaluate(pModel: UNet2D, pDataLoader: DataLoader, pCriterion: BCEWithLogitsLoss, pLog: YoonNLM):
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
    dTotalLoss = 0
    dTotalAcc = 0
    for i, (pTensorInput, pTensorTarget) in pBar:
        # Move data and label to device
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorInput.type(torch.FloatTensor).to(pDevice)
        # Pass the input data through the defined network architecture
        pTensorOutput = pModel(pTensorInput)
        # Compute a loss function
        pLoss = pCriterion(pTensorOutput, pTensorTarget)
        dTotalLoss += pLoss.item() * len(pTensorTarget[0])  # Loss per batch * batch
        # Compute network accuracy
        nAcc = torch.sum(torch.eq(pTensorOutput > 0.5, pTensorTarget > 0.5)).item()  # output and targets binary
        nLengthSample += len(pTensorInput[0])
        dTotalAcc += nAcc
        # Trace the log
        strMessage = pLog.write(i, len(pDataLoader),
                                CELoss=dTotalLoss / nLengthSample, ACC=(dTotalAcc / nLengthSample) * 100.0)
        pBar.set_description(strMessage)
    return dTotalLoss / nLengthSample


def train(nEpoch: int,
          strModelPath: str,
          pTrainData: YoonDataset,
          pTrainLabel: YoonDataset,
          pEvalData: YoonDataset,
          pEvalLabel: YoonDataset,
          nChannel=8,
          nCountDepth=4,
          nBatchSize=1,
          nCountWorker=2,  # 0: CPU / 2 : GPU
          dRateDropout=0.3,
          dRatioDecay=0.5,
          bInitEpoch=False):
    def learning_func(iStep):
        return 1.0 - max(0, iStep - nEpoch * (1 - dRatioDecay)) / (dRatioDecay * nEpoch + 1)

    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define the training and testing data-set
    pTrainSet = UNetDataset(pTrainData, pTrainLabel, dRatioTrain=0.8, strMode="Train")
    pTrainLoader = DataLoader(pTrainSet, batch_size=nBatchSize, shuffle=True, num_workers=nCountWorker, pin_memory=True)
    pValidationSet = UNetDataset(pEvalData, pEvalLabel, dRatioTrain=0.8, strMode="Eval")
    pValidationLoader = DataLoader(pValidationSet, batch_size=nBatchSize, shuffle=False,
                                   num_workers=nCountWorker, pin_memory=True)
    # Define a network model
    pModel = UNet2D(nDimInput=pTrainSet.input_dim, nDimOutput=pTrainSet.output_dim, nChannel=nChannel,
                    nCountDepth=nCountDepth, dRateDropout=dRateDropout).to(pDevice)
    # Set the optimizer with adam
    pOptimizer = torch.optim.Adam(pModel.parameters())
    # Set the training criterion
    pCriterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    # Set the scheduler to control the learning rate
    pScheduler = torch.optim.lr_scheduler.LambdaLR(pOptimizer, lr_lambda=learning_func)
    # Load pre-trained model
    nStart = 0
    print("Directory of the pre-trained model: {}".format(strModelPath))
    if strModelPath is not None and os.path.exists(strModelPath) and bInitEpoch is False:
        pModelData = torch.load(strModelPath, map_location=pDevice)
        nStart = pModelData['epoch']
        pModel.load_state_dict(pModelData['model'])
        pOptimizer.load_state_dict(pModelData['optimizer'])
        print("## Successfully load the model at {} epochs!".format(nStart))
    # Define the log manager
    pNLMTrain = YoonNLM(nStart, strRoot="./NLM/UNet2D", strMode="Train")
    pNLMEval = YoonNLM(nStart, strRoot="./NLM/UNet2D", strMode="Eval")
    # Train and Test Repeat
    dMinLoss = 10000.0
    for iEpoch in range(nStart, nEpoch + 1):
        # Train the network
        __process_train(pModel=pModel, pDataLoader=pTrainLoader, pCriterion=pCriterion,
                        pOptimizer=pOptimizer, pLog=pNLMTrain)
        # Test the network
        dLoss = __process_evaluate(pModel=pModel, pDataLoader=pValidationLoader, pCriterion=pCriterion, pLog=pNLMEval)
        # Change the learning rate
        pScheduler.step()
        # Rollback the model when loss is NaN
        if math.isnan(dLoss):
            if strModelPath is not None and os.path.exists(strModelPath):
                # Reload the best model and decrease the learning rate
                pModelData = torch.load(strModelPath, map_location=pDevice)
                pModel.load_state_dict(pModelData['model'])
                pOptimizerData = pModelData['optimizer']
                pOptimizerData['param_groups'][0]['lr'] /= 2  # Decrease the learning rate by 2
                pOptimizer.load_state_dict(pOptimizerData)
                print("## Rollback the Model with half learning rate!")
        # Save the optimal model
        elif dLoss < dMinLoss:
            dMinLoss = dLoss
            torch.save({'epoch': iEpoch, 'model': pModel.state_dict(), 'optimizer': pOptimizer.state_dict()},
                       strModelPath)
        elif iEpoch % 100 == 0:
            torch.save({'epoch': iEpoch, 'model': pModel.state_dict(), 'optimizer': pOptimizer.state_dict()},
                       'unet_{}epoch.pth'.format(iEpoch))


def test(pTestData: YoonDataset,
         strModelPath: str,
         nChannel=8,
         nCountDepth=4,
         nCountWorker=2,  # 0: CPU / 2 : GPU
         dRateDropout=0.3):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define a data path for plot for test
    pDataSet = UNetDataset(pTestData)
    pDataLoader = DataLoader(pDataSet, batch_size=1, shuffle=False, num_workers=nCountWorker, pin_memory=True)
    # Load UNET model
    pModel = UNet2D(nDimInput=pDataSet.input_dim, nDimOutput=pDataSet.output_dim, nChannel=nChannel,
                    nCountDepth=nCountDepth, dRateDropout=dRateDropout).to(pDevice)
    pModel.eval()
    pFile = torch.load(strModelPath)
    pModel.load_state_dict(pFile['model'])
    print("Successfully load the Model in path")
    # Start the test sequence
    pBar = tqdm(pDataLoader)
    print("Length of data = ", len(pBar))
    pListOutput = []
    for i, pTensorInput in enumerate(pBar):
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorOutput = pModel(pTensorInput)
        pListOutput.append(pTensorOutput.detach().cpu().numpy())
    # Warp the tensor to Dataset
    return YoonDataset.from_tensor(pImage=numpy.concatenate(pListOutput))
