import os.path

import numpy
import torch
import torch.nn
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


class UNet2D(Module):
    def __init__(self):
        super(UNet2D, self).__init__()
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

    def forward(self, pTensorX: tensor):
        # Normalize input features (zero mean and unit variance).
        pXMean = torch.mean(pTensorX, dim=(1, -1))  # Mean of All pixels (sum / (x,y))
        pXStd = torch.std(pTensorX, dim=(1, -1))  # Std of All pixels (sum / (x,y))
        pXStd[pXStd < 0.01] = 0.01
        pTensorX = (pTensorX - pXMean[:, None, None]) / pXStd[:, None, None]
        # Down-sampling
        pEncoder1 = self.encoder1(pTensorX)
        pEncoder2 = self.encoder2(self.down_sampler1(pEncoder1))
        pEncoder3 = self.encoder3(self.down_sampler2(pEncoder2))
        pEncoder4 = self.encoder4(self.down_sampler3(pEncoder3))
        pEncoder5 = self.encoder5(self.down_sampler4(pEncoder4))
        # Up-sampling
        pDecoder5 = self.decoder5(pEncoder5)
        pTensorChain4 = torch.cat((self.up_sampler4(pDecoder5), pEncoder4), dim=1)  # (0: batch dir, 1: CH dir ...)
        pDecoder4 = self.decoder4(pTensorChain4)
        pTensorChain3 = torch.cat((self.up_sampler3(pDecoder4), pEncoder3), dim=1)
        pDecoder3 = self.decoder3(pTensorChain3)
        pTensorChain2 = torch.cat((self.up_sampler2(pDecoder3), pEncoder2), dim=1)
        pDecoder2 = self.decoder2(pTensorChain2)
        pTensorChain1 = torch.cat((self.up_sampler1(pDecoder2), pEncoder1), dim=1)
        pDecoder1 = self.decoder1(pTensorChain1)
        # Full-convolution Layer
        pTensorResult = self.fc_layer(pDecoder1)
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
def __process_train(nEpoch: int, pModel: UNet2D, pDataLoader: DataLoader, pCriterion: BCEWithLogitsLoss,
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
def __process_test(pModel: UNet2D, pDataLoader: DataLoader, pCriterion: BCEWithLogitsLoss,
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
    pModel = UNet2D().to(pDevice)
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
        dLoss = __process_test(pModel=pModel, pDataLoader=pValidationLoader, pCriterion=pCriterion)
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
    pModel = UNet2D().to(pDevice)
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
