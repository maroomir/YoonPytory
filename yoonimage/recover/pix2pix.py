import math
import os.path
from datetime import datetime

import h5py
import numpy
import scipy
import scipy.io
import skimage.metrics
import torch
import torch.nn
import torch.nn.functional
from numpy import ndarray
from torch import tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomTransform(object):
    def __init__(self,
                 dMean=0.5,
                 dStd=0.5):
        self.mean = dMean
        self.std = dStd

    def __call__(self,
                 pImage: ndarray,
                 pTarget: ndarray = None):
        def to_tensor(pArray: ndarray):  # Height X Width X Channel
            if numpy.iscomplexobj(pArray):
                pArray = numpy.stack((pArray.real, pArray.imag), axis=-1)
            pArray = pArray.transpose((2, 0, 1)).astype(numpy.float32)  # Channel X Height X Width
            return torch.from_numpy(pArray)

        def z_normalize(pTensor: tensor):
            pFuncMask = pTensor > 0
            pMean = pTensor[pFuncMask].mean()
            pStd = pTensor[pFuncMask].std()
            return torch.mul((pTensor - pMean) / pStd, pFuncMask.float())

        def minmax_normalize(pTensor: tensor):
            pMax = pTensor.max()
            pMin = pTensor.min()
            return (pTensor - pMin) / (pMax - pMin)

        def normalize(pTensor: tensor):
            return (pTensor - self.mean) / self.std

        def pixel_compress(pTensor: tensor):
            pFuncMax = pTensor > 255
            pFuncMin = pTensor < 0
            pTensor[pFuncMax] = 255
            pTensor[pFuncMin] = 0
            return pTensor / 255.0

        pTensorInput = minmax_normalize(to_tensor(pImage))
        if pTarget is not None:
            pTensorTarget = minmax_normalize(to_tensor(pTarget))
            return pTensorInput, pTensorTarget
        else:
            return pTensorInput


class ConvDataset(Dataset):
    def __init__(self,
                 strFilePath,
                 pTransform=None,
                 strMode="train",  # train, eval, test
                 dTrainRatio=0.8):
        self.transform = pTransform
        self.mode = strMode
        # Initial the H5PY Inputs
        self.len = 0
        self.height = 0
        self.width = 0
        self.channel = 0
        self.input_data = None
        self.label_data = None
        self.load_dataset(strFilePath, dRatio=dTrainRatio)

    def load_dataset(self,
                     strFilePath: str,
                     dRatio: float):
        pFile = h5py.File(strFilePath)
        pInputData = numpy.array(pFile['input'], dtype=numpy.float32)  # 216 X 384 X 384 X 3
        try:
            pLabelData = numpy.array(pFile['label'], dtype=numpy.float32)  # 216 X 384 X 384 X 1
        except:
            pLabelData = None
            print("Label data is not contained")
        self.len, self.height, self.width, self.channel = pInputData.shape
        nCountTrain = int(self.len * dRatio)
        if self.mode == "train":
            self.input_data = pInputData[:nCountTrain, :, :, :]
            self.label_data = pLabelData[:nCountTrain, :, :, :]
        elif self.mode == "eval":
            self.input_data = pInputData[nCountTrain:, :, :, :]
            self.label_data = pLabelData[nCountTrain:, :, :, :]
        elif self.mode == "test":
            self.input_data = pInputData
        else:
            raise Exception("Data mode is not compatible")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, item):
        if self.mode == "test":
            return self.transform(self.input_data[item])
        else:
            return self.transform(self.input_data[item], self.label_data[item])


class CustomLoss(Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.prediction = None
        self.target = None
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pTensorPredict: tensor, pTensorTarget: tensor):
        pTensorLossMSE = self.mse_loss(pTensorPredict, pTensorTarget)
        pTensorLossL1 = self.l1_loss(pTensorPredict, pTensorTarget)
        self.prediction = pTensorPredict.detach().cpu()
        self.target = pTensorTarget.detach().cpu()
        dSSIM = 1 - self.ssim_score()
        return pTensorLossMSE + pTensorLossL1 + dSSIM

    def psnr_score(self):
        dPSNR = 0.0
        for iBatch in range(self.prediction.shape[0]):
            pArrayPredict = self.prediction[iBatch].numpy()
            pArrayTarget = self.target[iBatch].numpy()
            dPSNR += skimage.metrics.peak_signal_noise_ratio(pArrayTarget, pArrayPredict)
        return dPSNR / self.prediction.shape[0]

    def ssim_score(self):
        dSSIM = 0.0
        for iBatch in range(self.prediction.shape[0]):
            pArrayPredict = self.prediction[iBatch].numpy()
            pArrayTarget = self.target[iBatch].numpy()
            dSSIM += skimage.metrics.structural_similarity(pArrayTarget, pArrayPredict)
        return dSSIM / self.prediction.shape[0]

    def dice_coefficient(self, dSmooth=1e-4):
        pTensorPredict = self.prediction.contiguous().view(-1)
        pTensorTarget = self.target.contiguous().view(-1)
        pTensorIntersection = (pTensorPredict * pTensorTarget).sum()
        pTensorCoefficient = (2.0 * pTensorIntersection + dSmooth) / (
                pTensorPredict.sum() + pTensorTarget.sum() + dSmooth)
        return pTensorCoefficient


def save_labels(pArrayOutput: ndarray, strFilePath):
    scipy.io.savemat(strFilePath, mdict={'y_pred': pArrayOutput})
    print("Save output files completed!")


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


class DiscriminateBlock(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int):
        super(DiscriminateBlock, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(nDimInput, nDimOutput, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(nDimOutput),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, pTensorX: tensor):
        return self.network(pTensorX)


class Discriminator(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 nChannel: int,
                 nCountDepth: int):
        super(Discriminator, self).__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(nDimInput, nChannel, kernel_size=4, stride=2, padding=1),
                                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))])
        for i in range(nCountDepth - 1):
            self.blocks += [DiscriminateBlock(nChannel, nChannel * 2)]
            nChannel *= 2
        self.blocks += [torch.nn.Sequential(
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(nChannel, nDimOutput, kernel_size=4, stride=2, padding=1, bias=False))]

    def forward(self,
                pTensorInput: tensor,
                pTensorTarget: tensor):
        pTensorResult = torch.cat((pTensorInput, pTensorTarget), dim=1)
        for i, pBlock in enumerate(self.blocks):
            pTensorResult = pBlock(pTensorResult)
        return pTensorResult


class GeneratorUNet(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 nChannel: int,
                 nCountDepth: int,
                 dRateDropout: float = 0.3):
        super(GeneratorUNet, self).__init__()
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


def __trace_in__(strMessage: str):
    pNow = datetime.now()
    strFilePath = "LOG_PIX2PIX_" + pNow.strftime("%Y-%m-%d") + '.txt'
    with open(strFilePath, mode='a') as pFile:
        pFile.write("[" + pNow.strftime("%H:%M:%S") + "] " + strMessage + "\n")


def __process_train(nEpoch: int, pDataLoader: DataLoader,
                    pGenerator: GeneratorUNet, pDiscriminator: Discriminator,
                    pCriterionGenerate: CustomLoss, pCriterionDiscrimanate: Module,
                    pOptimizerGenerate: Adam, pOptimizerDiscriminate: Adam):
    def __train_generator(pInput: tensor,
                          pTarget: tensor,
                          dRateOutside=0.01):
        pOptimizerGenerate.zero_grad()
        pTargetFake = pGenerator(pInput).to(pDevice)
        pPredictFake = pDiscriminator(pInput, pTargetFake).to(pDevice)
        # Reshape the tensor data of use
        pTargetFake = pTargetFake.squeeze(axis=1)  # reshape : (batch, 384, 384)
        pPredictFake = pPredictFake.squeeze(axis=1)
        pTarget = pTarget.squeeze(axis=1)
        # Set-up the ground truths (Shape like the prediction size)
        pPass = torch.ones(pPredictFake.size(), requires_grad=True).type(torch.FloatTensor).to(pDevice)
        # Compute the network accuracy
        pLossGenerator = pCriterionGenerate(pTargetFake, pTarget)  # Loss the generator
        pLossDiscriminator = pCriterionDiscrimanate(pPredictFake, pPass)  # Loss the pass rate of the fake image
        pLoss = pLossGenerator + (pLossDiscriminator * dRateOutside)
        dPSNR = pCriterionGenerate.psnr_score()
        dSSIM = pCriterionGenerate.ssim_score()
        # Perform backpropagation to update GENERATOR parameters
        pLoss.backward()
        pOptimizerGenerate.step()
        # Fix the CUDA Out of Memory problem
        del pTargetFake
        del pPredictFake
        torch.cuda.empty_cache()
        return pLoss.item(), dPSNR, dSSIM

    def __train_discriminator(pInput: tensor,
                              pTarget: tensor):
        pOptimizerDiscriminate.zero_grad()
        pTargetFake = pGenerator(pInput).to(pDevice).to(pDevice)
        pPredictReal = pDiscriminator(pInput, pTarget).to(pDevice)
        pPredictFake = pDiscriminator(pInput, pTargetFake).to(pDevice)
        # Reshape the tensor data of use
        pPredictReal = pPredictReal.squeeze(axis=1)
        pPredictFake = pPredictFake.squeeze(axis=1)
        # Set-up the ground truths (Shape like the prediction size)
        pPass = torch.ones(pPredictReal.size(), requires_grad=True).type(torch.FloatTensor).to(pDevice)
        pNG = torch.zeros(pPredictReal.size(), requires_grad=True).type(torch.FloatTensor).to(pDevice)
        # Compute the Real Pass or NG rate
        pLossRealNG = pCriterionDiscrimanate(pPredictFake, pNG)
        pLossRealOK = pCriterionDiscrimanate(pPredictReal, pPass)
        pLoss = pLossRealNG + pLossRealOK
        # Perform backpropagation to update GENERATOR parameters
        pLoss.backward()
        pOptimizerDiscriminate.step()
        # Fix the CUDA Out of Memory problem
        del pTargetFake
        del pPredictReal
        del pPredictFake
        torch.cuda.empty_cache()
        return pLoss.item(), pLossRealOK.item(), pLossRealNG.item()

    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Perform a training using the defined network
    pGenerator.train()
    pDiscriminator.train()
    # Warp the iterable Data Loader with TQDM
    pBar = tqdm(enumerate(pDataLoader))
    strMessage = ""
    for i, (pTensorInput, pTensorTarget) in pBar:
        # Move data and label to device
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)  # Shape : (batch, 3, 384, 384)
        pTensorTarget = pTensorTarget.type(torch.FloatTensor).to(pDevice)  # Shape : (batch, 1, 384, 384)
        # Pass the input data through the GENERATOR architecture
        dLossGenerator, dPSNR, dSSIM = __train_generator(pTensorInput, pTensorTarget, dRateOutside=0.01)
        # Pass the input data through the GENERATOR architecture
        dLossDiscriminator, dRealOK, dRealNG = __train_discriminator(pTensorInput, pTensorTarget)
        # Perform backpropagation to update DISCRIMINATOR parameters
        strMessage = "Train Epoch:{:3d} [{}/{} {:.2f}%], " \
                     "Generator={:.4f}, PSNR={:.4f}, SSIM={:.4f}, " \
                     "RealOK={:.4f}, RealNG={:.4f}".format(nEpoch, i + 1, len(pDataLoader),
                                                           100.0 * ((i + 1) / len(pDataLoader)),
                                                           dLossGenerator, dPSNR, dSSIM, dRealOK, dRealNG)
        pBar.set_description(strMessage)
    # trace the last message
    __trace_in__(strMessage)


def __process_evaluate(pModel: GeneratorUNet, pDataLoader: DataLoader, pCriterion: CustomLoss):
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
    dTotalPSNR = 0.0
    dTotalSSIM = 0.0
    strMessage = ""
    with torch.no_grad():
        for i, (pTensorInput, pTensorTarget) in pBar:
            # Move data and label to device
            pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
            pTensorTarget = pTensorTarget.type(torch.FloatTensor).to(pDevice)
            # Pass the input data through the defined network architecture
            pTensorOutput = pModel(pTensorInput).squeeze(axis=1)  # reshape : (batch, 384, 384)
            pTensorTarget = pTensorTarget.squeeze(axis=1)  # reshape : (batch, 384, 384)
            # Compute a loss function
            pTensorLoss = pCriterion(pTensorOutput, pTensorTarget).to(pDevice)
            nTotalLoss += pTensorLoss.item() * len(pTensorTarget)
            # Compute network accuracy
            dTotalPSNR += pCriterion.psnr_score() * len(pTensorTarget)
            dTotalSSIM += pCriterion.ssim_score() * len(pTensorTarget)
            nLengthSample += len(pTensorTarget)
            strMessage = "Eval {}/{} {:.2f}%, Loss={:.4f}, PSNR={:.4f}, SSIM={:.4f}". \
                format(i + 1, len(pDataLoader), 100.0 * ((i + 1) / len(pDataLoader)),
                       nTotalLoss / nLengthSample, dTotalPSNR / nLengthSample, dTotalSSIM / nLengthSample)
            pBar.set_description(strMessage)
    # trace the last message
    __trace_in__(strMessage)
    # Fix the CUDA Out of Memory problem
    del pTensorOutput
    del pTensorLoss
    torch.cuda.empty_cache()
    return nTotalLoss / nLengthSample


def train(nEpoch: int,
          strPath: str,
          strGeneratorPath: str = None,
          strDiscriminatorPath: str = None,
          nChannel=8,
          nCountDepth=4,
          nBatchSize=1,
          nCountWorker=2,  # 0: CPU / 2 : GPU
          dRateDropout=0.3,
          dRatioDecay=0.5,
          bInitEpoch=False,
          ):

    def learning_func(iStep):
        return 1.0 - max(0, iStep - nEpoch * (1 - dRatioDecay)) / (dRatioDecay * nEpoch + 1)

    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define the training and testing data-set
    pTrainSet = ConvDataset(strFilePath=strPath, pTransform=CustomTransform(), strMode="train", dTrainRatio=0.8)
    pTrainLoader = DataLoader(dataset=pTrainSet, batch_size=nBatchSize, shuffle=True,
                              num_workers=nCountWorker, pin_memory=True)
    pValidationSet = ConvDataset(strFilePath=strPath, pTransform=CustomTransform(), strMode="eval", dTrainRatio=0.8)
    pValidationLoader = DataLoader(dataset=pValidationSet, batch_size=1, shuffle=False,
                                   num_workers=nCountWorker, pin_memory=True)
    # Define a network model
    pGenerator = GeneratorUNet(nDimInput=3, nDimOutput=1, nChannel=nChannel, nCountDepth=nCountDepth,
                               dRateDropout=dRateDropout).to(pDevice)  # T1, T2, GRE(3) => STIR(1)
    pDiscriminator = Discriminator(nDimInput=4, nDimOutput=1, nChannel=64,
                                   nCountDepth=nCountDepth).to(pDevice)  # Input(3), Target(1) => BOOL(1)
    # Set the optimizer with adam
    pOptimizerGenerate = torch.optim.Adam(pGenerator.parameters())
    pOptimizerDiscriminate = torch.optim.Adam(pDiscriminator.parameters())
    # Set the scheduler to control the learning rate
    pSchedulerGenerate = torch.optim.lr_scheduler.LambdaLR(pOptimizerGenerate, lr_lambda=learning_func)
    pSchedulerDiscriminate = torch.optim.lr_scheduler.LambdaLR(pOptimizerDiscriminate, lr_lambda=learning_func)
    # Set the Loss Function
    pCriterionGenerator = CustomLoss()
    pCriterionDiscriminate = torch.nn.MSELoss()
    # Load pre-trained model
    nStart = 0
    print("Directory of the generator model: {}".format(strGeneratorPath))
    if strGeneratorPath is not None and os.path.exists(strGeneratorPath) and bInitEpoch is False:
        pModelData = torch.load(strGeneratorPath, map_location=pDevice)
        nStart = pModelData['epoch']
        pGenerator.load_state_dict(pModelData['model'])
        pOptimizerGenerate.load_state_dict(pModelData['optimizer'])
        print("## Successfully load the Generator!")
    print("Directory of the discriminator model: {}".format(strDiscriminatorPath))
    if strDiscriminatorPath is not None and os.path.exists(strDiscriminatorPath) and bInitEpoch is False:
        pModelData = torch.load(strDiscriminatorPath, map_location=pDevice)
        nStart = pModelData['epoch'] if pModelData['epoch'] < nStart else nStart
        pDiscriminator.load_state_dict(pModelData['model'])
        pOptimizerDiscriminate.load_state_dict(pModelData['optimizer'])
        print("## Successfully load the Discriminator!")
    # Train and Test Repeat
    dMinLoss = 10000.0
    for iEpoch in range(nStart, nEpoch + 1):
        # Train the network
        __process_train(iEpoch, pDataLoader=pTrainLoader, pGenerator=pGenerator, pDiscriminator=pDiscriminator,
                        pCriterionGenerate=pCriterionGenerator, pCriterionDiscrimanate=pCriterionDiscriminate,
                        pOptimizerGenerate=pOptimizerGenerate, pOptimizerDiscriminate=pOptimizerDiscriminate)
        # Test the network
        dLoss = __process_evaluate(pModel=pGenerator, pDataLoader=pValidationLoader, pCriterion=pCriterionGenerator)
        # Change the learning rate
        pSchedulerGenerate.step()
        pSchedulerDiscriminate.step()
        # Rollback the model when loss is NaN
        if math.isnan(dLoss):
            if strGeneratorPath is not None and os.path.exists(strGeneratorPath):
                # Reload the best model and decrease the learning rate
                pModelData = torch.load(strGeneratorPath, map_location=pDevice)
                pGenerator.load_state_dict(pModelData['model'])
                pOptimizerData = pModelData['optimizer']
                pOptimizerData['param_groups'][0]['lr'] /= 2  # Decrease the learning rate by 2
                pOptimizerGenerate.load_state_dict(pOptimizerData)
                print("## Rollback the Generator with half learning rate!")
            if strDiscriminatorPath is not None and os.path.exists(strDiscriminatorPath):
                # Reload the best model and decrease the learning rate
                pModelData = torch.load(strDiscriminatorPath, map_location=pDevice)
                pDiscriminator.load_state_dict(pModelData['model'])
                pOptimizerData = pModelData['optimizer']
                pOptimizerData['param_groups'][0]['lr'] /= 2  # Decrease the learning rate by 2
                pOptimizerDiscriminate.load_state_dict(pOptimizerData)
                print("## Rollback the Discriminator with half learning rate!")
        # Save the optimal model
        elif dLoss < dMinLoss:
            dMinLoss = dLoss
            torch.save({'epoch': iEpoch, 'model': pGenerator.state_dict(),
                        'optimizer': pOptimizerGenerate.state_dict()}, strGeneratorPath)
            torch.save({'epoch': iEpoch, 'model': pDiscriminator.state_dict(),
                        'optimizer': pOptimizerDiscriminate.state_dict()}, strDiscriminatorPath)
        elif iEpoch % 100 == 0:
            torch.save({'epoch': iEpoch, 'model': pGenerator.state_dict(),
                        'optimizer': pOptimizerGenerate.state_dict()}, 'gen_{}epoch.pth'.format(iEpoch))
            torch.save({'epoch': iEpoch, 'model': pDiscriminator.state_dict(),
                        'optimizer': pOptimizerDiscriminate.state_dict()}, 'disc_{}epoch.pth'.format(iEpoch))


def test(strPath: str,
         strModelPath: str,
         nChannel=8,
         nCountDepth=4,
         nCountWorker=2,  # 0: CPU / 2 : GPU
         dRateDropout=0.3):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define a network model
    pModel = GeneratorUNet(nDimInput=3, nDimOutput=1, nChannel=nChannel, nCountDepth=nCountDepth,
                           dRateDropout=dRateDropout).to(pDevice)
    pModelData = torch.load(strModelPath, map_location=pDevice)
    pModel.load_state_dict(pModelData['model'])
    pModel.eval()
    print("Successfully load the Model in path")
    # Define the validation data-set
    pTestSet = ConvDataset(strFilePath=strPath, pTransform=CustomTransform(), strMode="test")
    pTestLoader = DataLoader(dataset=pTestSet, batch_size=1, shuffle=False,
                             num_workers=nCountWorker, pin_memory=True)
    pBar = tqdm(pTestLoader)
    pListResult = []
    with torch.no_grad():
        for pTensorInput in pBar:
            pTensorInput = pTensorInput.to(pDevice)
            pTensorResult = pModel(pTensorInput)
            pTensorResult = torch.argmax(pTensorResult, dim=1)
            for i in range(pTensorResult.shape[0]):  # Attach per batch
                pListResult.append(pTensorResult[i].detach().cpu().numpy())
    # Save the result to mat
    save_labels(numpy.array(pListResult), '2021451143_CheoljoungYoon_ContrastConversion.mat')


if __name__ == '__main__':
    mode = 'train'
    if mode == 'all':
        train(nEpoch=1000,
              strPath='contrast_conversion_train_dataset.mat',
              strGeneratorPath='model_gen.pth',
              strDiscriminatorPath='model_disc.pth',
              nChannel=64,  # 8 >= VRAM 9GB / 4 >= VRAM 6.5GB
              nCountDepth=4,
              nBatchSize=2,
              nCountWorker=2,  # 0= CPU / 2 >= GPU
              dRateDropout=0.3,
              dRatioDecay=0.5,
              bInitEpoch=False)
        test(strPath='contrast_conversion_train_dataset.mat',
             strModelPath='model_gen.pth',
             nChannel=64,  # 8 : colab / 4 : RTX2070
             nCountDepth=4,
             nCountWorker=2,  # 0: CPU / 2 : GPU
             dRateDropout=0)
    elif mode == 'train':
        train(nEpoch=3000,
              strPath='contrast_conversion_train_dataset.mat',
              strGeneratorPath='model_gen.pth',
              strDiscriminatorPath='model_disc.pth',
              nChannel=64,  # 8 >= VRAM 9GB / 4 >= VRAM 6.5GB
              nCountDepth=4,
              nBatchSize=2,
              nCountWorker=2,  # 0= CPU / 2 >= GPU
              dRateDropout=0.3,
              dRatioDecay=0.5,
              bInitEpoch=False)
    elif mode == 'test':
        test(strPath='contrast_conversion_train_dataset.mat',
             strModelPath='model_gen.pth',
             nChannel=64,  # 8 : colab / 4 : RTX2070
             nCountDepth=4,
             nCountWorker=2,  # 0: CPU / 2 : GPU
             dRateDropout=0)
    else:
        pass
