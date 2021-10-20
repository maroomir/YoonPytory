import math
import os

import matplotlib.pyplot
import numpy.random
import sklearn.metrics
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from yoonimage.classification.dataset import ClassificationDataset, collate_segmentation
from yoonimage.data import YoonDataset, YoonTransform
from yoonpytory.log import YoonNLM


class VGG(Module):
    __config_dict = {
        "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self,
                 nDimInput,
                 nNumClass,
                 strType="VGG16"):
        super(VGG, self).__init__()
        self.channel = nDimInput
        self.network = self.__make_layers(self.__config_dict[strType])
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(512 * 1 * 1, 360),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(360, 100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(100, nNumClass),
        )

    def forward(self, pTensorX: Tensor):
        pTensorOutput = self.network(pTensorX)
        pTensorOutput = pTensorOutput.view(pTensorOutput.size(0), -1)
        pTensorOutput = self.fc_layer(pTensorOutput)
        return pTensorOutput

    def __make_layers(self, pListConfig):
        pListLayer = []
        nChannel = self.channel
        for pParam in pListConfig:
            if pParam == 'M':
                pListLayer += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                pListLayer += [torch.nn.Conv2d(in_channels=nChannel, out_channels=pParam, kernel_size=3, padding=1),
                               torch.nn.BatchNorm2d(pParam),
                               torch.nn.ReLU(inplace=True)]
                nChannel = pParam
        return torch.nn.Sequential(*pListLayer)


def __process_train(pModel: VGG, pDataLoader: DataLoader, pOptimizer, pCriterion, pLog: YoonNLM):
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
                                Loss=dTotalLoss / nLengthSample, Acc=100 * nTotalCorrect / nLengthSample)
        pBar.set_description(strMessage)


def __process_evaluate(pModel: VGG,
                       pDataLoader: DataLoader,
                       pCriterion,
                       pLog: YoonNLM):
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
                                    Loss=dTotalLoss / nLengthSample, Acc=100 * nTotalCorrect / nLengthSample)
            pBar.set_description(strMessage)
    return dTotalLoss / nLengthSample


def __process_test(pModel: VGG, pDataLoader: DataLoader, pListLabel: list):
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
          pTransform: YoonTransform,
          strModelMode="VGG19",
          nBatchSize=32,
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
    pTrainSet = ClassificationDataset(pTrainData, nCountClass, pTransform)
    pTrainLoader = DataLoader(pTrainSet, batch_size=nBatchSize, shuffle=True,
                              collate_fn=collate_segmentation, num_workers=nCountWorker, pin_memory=True)
    pValidationSet = ClassificationDataset(pEvalData, nCountClass, pTransform)
    pValidationLoader = DataLoader(pValidationSet, batch_size=nBatchSize, shuffle=False,
                                   collate_fn=collate_segmentation, num_workers=nCountWorker, pin_memory=True)
    # Define a network model
    pModel = VGG(nDimInput=pTrainSet.input_dim, nNumClass=pTrainSet.output_dim, strType=strModelMode).to(pDevice)
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
    pNLMTrain = YoonNLM(nStart, strRoot="./NLM/VGG", strMode="Train")
    pNLMEval = YoonNLM(nStart, strRoot="./NLM/VGG", strMode="Eval")
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
                       'vgg_{}epoch.pth'.format(iEpoch))


def test(pTestData: YoonDataset,
         strModelPath: str,
         nCountClass: int,
         pTransform: YoonTransform,
         strModelMode="VGG19",
         nCountWorker=0,  # 0: CPU / 4 : GPU
         ):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define a data path for plot for test
    pDataSet = ClassificationDataset(pTestData, nCountClass, pTransform)
    pDataLoader = DataLoader(pDataSet, batch_size=1, shuffle=False,
                             collate_fn=collate_segmentation, num_workers=nCountWorker, pin_memory=True)
    # Load the model
    pModel = VGG(nDimInput=pDataSet.input_dim, nNumClass=pDataSet.output_dim, strType=strModelMode).to(pDevice)
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
    return YoonDataset.from_tensor(images=None, labels=numpy.concatenate(pListLabel))
