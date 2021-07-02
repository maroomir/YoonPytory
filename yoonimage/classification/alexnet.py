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


class AlexNet(Module):
    def __init__(self,
                 nDimInput,
                 nNumClass):
        super(AlexNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(nDimInput, 96, kernel_size=11, stride=4, padding=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(256 * 2 * 2, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, nNumClass)
        )

    def forward(self, pTensorX: Tensor):
        pTensorX = self.network(pTensorX)
        pTensorX = pTensorX.view(pTensorX.size(0), -1)
        pTensorX = self.fc_layer(pTensorX)
        return pTensorX


def __process_train(pModel: AlexNet,
                    pDataLoader: DataLoader,
                    pOptimizer,
                    pCriterion,
                    pLog: YoonNLM):
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


def __process_evaluate(pModel: AlexNet,
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


def __process_test(pModel: AlexNet, pDataLoader: DataLoader, pListLabel: list):
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
    pModel = AlexNet(nDimInput=pTrainSet.input_dim, nNumClass=pTrainSet.output_dim).to(pDevice)
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
    pNLMTrain = YoonNLM(nStart, strRoot="./NLM/AlexNet", strMode="Train")
    pNLMEval = YoonNLM(nStart, strRoot="./NLM/AlexNet", strMode="Eval")
    # Train and Test Repeat
    dMinLoss = 10000.0
    for iEpoch in range(nStart, nEpoch + 1):
        __process_train(pModel=pModel, pDataLoader=pTrainLoader, pCriterion=pCriterion,
                        pOptimizer=pOptimizer, pLog=pNLMTrain)
        dLoss = __process_evaluate(pModel=pModel, pDataLoader=pValidationLoader, pCriterion=pCriterion,
                                   pLog=pNLMEval)
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
                       'alexnet_{}epoch.pth'.format(iEpoch))


def test(pTestData: YoonDataset,
         strModelPath: str,
         nCountClass: int,
         pTransform: YoonTransform,
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
    pDataLoader = DataLoader(pDataSet, batch_size=1, shuffle=False, collate_fn=collate_segmentation,
                             num_workers=nCountWorker, pin_memory=True)
    # Load the model
    pModel = AlexNet(nDimInput=pDataSet.input_dim, nNumClass=pDataSet.output_dim).to(pDevice)
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
