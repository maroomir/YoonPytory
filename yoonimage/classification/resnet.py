import os
import math

import numpy.random
import torch
from torch import tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot
import sklearn.metrics
from tqdm import tqdm

from yoonimage.data import YoonDataset
from yoonpytory.log import YoonNLM
from yoonimage.classification.dataset import ClassificationDataset, collate_segmentation


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
        super(IdentityBlock, self).__init__()
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
        pTensorOut = self.network(pTensorX)
        pTensorOut += pTensorX
        pTensorOut = torch.nn.functional.relu(pTensorOut)
        return pTensorOut


class ResNet50(Module):  # Conv Count = 50
    def __init__(self,
                 nDimInput: int,
                 nNumClass: int):
        super(ResNet50, self).__init__()
        self.layer1 = torch.nn.Sequential(  # Conv=1
            torch.nn.Conv2d(nDimInput, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.layer2 = torch.nn.Sequential(  # Conv=10
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvolutionBlock(nDimInput=64, pListFilter=[64, 64, 256], nStride=1),  # Conv=4
            IdentityBlock(pListFilter=[64, 64, 256]),  # Conv=3
            IdentityBlock(pListFilter=[64, 64, 256])  # Conv=3
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
        self.fc_layer = torch.nn.Linear(2048, nNumClass)

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


def __process_train(pModel: ResNet50, pDataLoader: DataLoader, pOptimizer, pCriterion, pLog: YoonNLM):
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    pModel.train()
    pBar = tqdm(enumerate(pDataLoader))
    dTotalLoss = 0.0
    nTotalCorrect = 0
    nLengthSample = 0
    for i, (pTensorInput, pTensorTarget) in pBar:
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.type(torch.LongTensor).to(pDevice)
        pTensorOutput = pModel(pTensorInput)
        pOptimizer.zero_grad()
        pTensorLoss = pCriterion(pTensorOutput, pTensorTarget)
        pTensorLoss.backward()
        pOptimizer.step()
        dTotalLoss += pTensorLoss.item() * pTensorTarget.size(0)
        _, pTensorPredicted = pTensorOutput.max(1)
        nLengthSample += pTensorTarget.size(0)
        nTotalCorrect += pTensorPredicted.eq(pTensorTarget).sum().item()
        strMessage = pLog.write(i, len(pDataLoader),
                                Loss=dTotalLoss/nLengthSample, Acc=100*nTotalCorrect/nLengthSample)
        pBar.set_description(strMessage)


def __process_evaluate(pModel: ResNet50,
                       pDataLoader: DataLoader,
                       pCriterion,
                       pLog: YoonNLM):
    print("\nTest: ")
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    pModel.eval()
    pBar = tqdm(enumerate(pDataLoader))
    dTotalLoss = 0.0
    nTotalCorrect = 0
    nLengthSample = 0
    with torch.no_grad():
        for i, (pTensorInput, pTensorTarget) in pBar:
            pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
            pTensorTarget = pTensorTarget.type(torch.LongTensor).to(pDevice)
            pTensorOutput = pModel(pTensorInput)
            pTensorLoss = pCriterion(pTensorOutput, pTensorTarget)
            dTotalLoss += pTensorLoss.item() * pTensorTarget.size(0)
            _, pTensorPredicted = pTensorOutput.max(1)
            nLengthSample += pTensorTarget.size(0)
            nTotalCorrect += pTensorPredicted.eq(pTensorTarget).sum().item()
            strMessage = pLog.write(i, len(pDataLoader),
                                    Loss=dTotalLoss/nLengthSample, Acc=100*nTotalCorrect/nLengthSample)
            pBar.set_description(strMessage)
    return dTotalLoss / nLengthSample


def __process_test(pModel: ResNet50, pDataLoader: DataLoader, pListLabel: list):
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    pModel.eval()
    pListTargetLabel = []
    pListPredictedLabel = []
    for i, (pTensorInput, pTensorTarget) in enumerate(pDataLoader):
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.type(torch.LongTensor).to(pDevice)
        pTensorOutput = pModel(pTensorInput)
        _, pTensorPredicted = pTensorOutput.max(1)
        pListTargetLabel = numpy.concatenate((pListTargetLabel, pTensorTarget.cpu().numpy()))
        pListPredictedLabel = numpy.concatenate((pListPredictedLabel, pTensorPredicted.cpu().numpy()))
    # Compute confusion matrix
    pMatrix = sklearn.metrics.confusion_matrix(pListTargetLabel, pListPredictedLabel)
    numpy.set_printoptions(precision=2)
    pArrayCorrected = (pListPredictedLabel == pListTargetLabel)
    dAcc = numpy.sum(pArrayCorrected * 1) / len(pArrayCorrected)
    print("Accuracy: %.5f" % dAcc)
    # Plot non-normalized confusion matrix
    matplotlib.pyplot.figure()
    __draw_confusion_matrix(pMatrix=pMatrix, pListLabel=pListLabel, strTitle="Confusion matrix, without normalization")
    # Plot non-normalized confusion matrix
    matplotlib.pyplot.figure()
    __draw_confusion_matrix(pMatrix=pMatrix, pListLabel=pListLabel, bNormalize=True,
                            strTitle="Confusion matrix, without normalization")
    matplotlib.pyplot.show()


def __draw_confusion_matrix(pMatrix: numpy,
                            pListLabel: list,
                            bNormalize=False,
                            pColorMap=matplotlib.pyplot.cm.Blues,
                            strTitle="Confusion Matrix"):
    if bNormalize:
        pMatrix = pMatrix.astype('float') / pMatrix.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    matplotlib.pyplot.imshow(pMatrix, interpolation='nearest', cmap=pColorMap)
    matplotlib.pyplot.title(strTitle)
    matplotlib.pyplot.colorbar()
    pArrayMark = numpy.arange(len(pListLabel))
    matplotlib.pyplot.xticks(pArrayMark, pListLabel, rotation=45)
    matplotlib.pyplot.yticks(pArrayMark, pListLabel)
    strFormat = '.2f' if bNormalize else 'd'
    dThreshold = pMatrix.max() / 2.
    for i in range(pMatrix.shape[0]):
        for j in range(pMatrix.shape[1]):
            matplotlib.pyplot.text(j, i, format(pMatrix[i, j], strFormat), horizontalalignment="center", color="white"
            if pMatrix[i, j] > dThreshold else "black")
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.ylabel("True label")
    matplotlib.pyplot.xlabel("Predicted label")


def __draw_dataset(pDataSet: Dataset, pListLabel: list, nCountShow=15):
    nCount = len(pDataSet)
    pFigure = matplotlib.pyplot.figure()
    pListRandom = numpy.random.randint(nCount, size=nCountShow)
    nCountHorz = int(nCountShow / 2)
    nCountVert = nCountShow - nCountHorz
    for i in range(nCountShow):
        pPlot = pFigure.add_subplot(nCountHorz, nCountVert, i + 1)
        pPlot.set_xticks([])
        pPlot.set_yticks([])
        pImage, nLabel = pDataSet[pListRandom[i]]
        pPlot.set_title("%s" % pListLabel[nLabel])
        pPlot.imshow(pImage)
    matplotlib.pyplot.show()


def train(nEpoch: int,
          strModelPath: str,
          nCountClass: int,
          pTrainData: YoonDataset,
          pEvalData: YoonDataset,
          nBatchSize=8,
          nCountWorker=0,  # 0: CPU / 4 : GPU
          dLearningRate=0.1,
          bInitEpoch=False):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define the training and testing data-set
    pTrainSet = ClassificationDataset(pTrainData, nCountClass, "resize", "rechannel", "z_norm")
    pTrainLoader = DataLoader(pTrainSet, batch_size=nBatchSize, shuffle=True,
                              collate_fn=collate_segmentation, num_workers=nCountWorker, pin_memory=True)
    pValidationSet = ClassificationDataset(pEvalData, nCountClass, "resize", "rechannel", "z_norm")
    pValidationLoader = DataLoader(pValidationSet, batch_size=nBatchSize, shuffle=False,
                                   collate_fn=collate_segmentation, num_workers=nCountWorker, pin_memory=True)
    # Define a network model
    pModel = ResNet50(nDimInput=pTrainSet.input_dim, nNumClass=pTrainSet.output_dim).to(pDevice)
    pCriterion = torch.nn.CrossEntropyLoss()
    pOptimizer = torch.optim.SGD(pModel.parameters(), lr=dLearningRate, momentum=0.9, weight_decay=5e-4)
    pScheduler = torch.optim.lr_scheduler.StepLR(pOptimizer, step_size=20, gamma=0.5)
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
    pNLMTrain = YoonNLM(nStart, strRoot="./NLM/ResNet", strMode="Train")
    pNLMEval = YoonNLM(nStart, strRoot="./NLM/ResNet", strMode="Eval")
    # Train and Test Repeat
    dMinLoss = 10000.0
    for iEpoch in range(nStart, nEpoch + 1):
        __process_train(pModel=pModel, pDataLoader=pTrainLoader, pCriterion=pCriterion,
                        pOptimizer=pOptimizer, pLog=pNLMTrain)
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
                       'resnet_{}epoch.pth'.format(iEpoch))


def test(pTestData: YoonDataset,
         strModelPath: str,
         nCountClass: int,
         nCountWorker=0,  # 0: CPU / 4 : GPU
         ):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define a data path for plot for test
    pDataSet = ClassificationDataset(pTestData, nCountClass, "resize", "rechannel", "z_norm")
    pDataLoader = DataLoader(pDataSet, batch_size=1, shuffle=False,
                             collate_fn=collate_segmentation, num_workers=nCountWorker, pin_memory=True)
    # Load the model
    pModel = ResNet50(nDimInput=pDataSet.input_dim, nNumClass=pDataSet.output_dim).to(pDevice)
    pModel.eval()
    pFile = torch.load(strModelPath)
    pModel.load_state_dict(pFile['model'])
    print("Successfully load the Model in path")
    # Start the test sequence
    pBar = tqdm(pDataLoader)
    print("Length of data = ", len(pBar))
    pListLabel = []
    for i, pTensorInput in enumerate(pBar):
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorOutput = pModel(pTensorInput)
        _, pTensorPredicted = pTensorOutput.max(1)
        pListLabel.append(pTensorPredicted.detach().cpu().numpy())
    # Warp the tensor to Dataset
    return YoonDataset.from_tensor(pImage=None, pLabel=numpy.concatenate(pListLabel))
