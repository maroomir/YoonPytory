import numpy
import torch
import torch.nn
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
    # Grouping batch
    pListInput = torch.FloatTensor(pListInput)
    pListTarget = torch.FloatTensor(pListTarget)
    return pListInput, pListTarget


# Define a train function
def __process_train(nEpoch: int, pModel: UNet, pDataLoader: DataLoader, pCriterion: BCEWithLogitsLoss,
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
        nTotalLoss += pLoss.item() * len(pTensorTarget[0])  # Count batch
        # Compute network accuracy
        nAcc = torch.sum(torch.eq(torch.argmax())) # [FIXING] DO NOT USE
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
def __process_test(pModel: UNet, pDataLoader: DataLoader, pCriterion: BCEWithLogitsLoss,
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
        nTotalLoss += pLoss.item() * len(pTensorTarget[0])  # Count batch
        # Compute
