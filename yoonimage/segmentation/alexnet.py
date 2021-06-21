import numpy.random
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms
import matplotlib.pyplot
import sklearn.metrics

from yoonimage.data import YoonDataset
from yoonpytory.log import YoonNLM


class SegmentationDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset,
                 nDimOutput: int
                 ):
        self.data = pDataset
        self.data.resize(strOption="min")
        self.data.rechannel(strOption="min")
        self.data.normalize(strOption="z")
        self.input_dim = self.data.min_channel()
        self.output_dim = nDimOutput

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        pArrayInput = self.data[item].image.copy_tensor()
        nTarget = self.data[item].label
        return pArrayInput, nTarget


class AlexNet(Module):
    def __init__(self,
                 nNumClass=10):
        super(AlexNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5),
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


def __process_train(nEpoch: int, pModel: AlexNet, pDataLoader: DataLoader, pOptimizer, pCriterion):
    print("\nEpoch: {}".format(nEpoch))
    print("\nTrain: ")
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    pModel.train()
    dLossTotal = 0.0
    nCountCorrect = 0
    nCountTotal = 0
    for i, (pTensorInput, pTensorTarget) in enumerate(pDataLoader):
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.type(torch.LongTensor).to(pDevice)
        pTensorOutput = pModel(pTensorInput)
        pOptimizer.zero_grad()
        pTensorLoss = pCriterion(pTensorOutput, pTensorTarget)
        pTensorLoss.backward()
        pOptimizer.step()
        dLossTotal += pTensorLoss.item()
        _, pTensorPredicted = pTensorOutput.max(1)
        nCountTotal += pTensorTarget.size(0)
        nCountCorrect += pTensorPredicted.eq(pTensorTarget).sum().item()

        if i + 1 == len(pDataLoader):
            print("[%3d/%3d] | Loss : %.3f | Acc : %.3f%% (%d/%d)" % (
                i + 1, len(pDataLoader), dLossTotal / (i + 1), 100. * nCountCorrect / nCountTotal,
                nCountCorrect, nCountTotal))


def __process_evaluate(iEpoch: int, pModel: AlexNet, pDataLoader: DataLoader, pCriterion, pScheduler):
    print("\nTest: ")
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    pModel.eval()
    dLossTotal = 0.0
    nCountCorrect = 0
    nCountTotal = 0
    with torch.no_grad():
        for i, (pTensorInput, pTensorTarget) in enumerate(pDataLoader):
            pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
            pTensorTarget = pTensorTarget.type(torch.LongTensor).to(pDevice)
            pTensorOutput = pModel(pTensorInput)
            pTensorLoss = pCriterion(pTensorOutput, pTensorTarget)
            dLossTotal += pTensorLoss.item()
            _, pTensorPredicted = pTensorOutput.max(1)
            nCountTotal += pTensorTarget.size(0)
            nCountCorrect += pTensorPredicted.eq(pTensorTarget).sum().item()
            if i + 1 == len(pDataLoader):
                print("[%3d/%3d] | Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                    i + 1, len(pDataLoader), dLossTotal / (i + 1), 100. * nCountCorrect / nCountTotal,
                    nCountCorrect, nCountTotal))
    pScheduler.step()


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


def main(nEpoch=50):
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Visualizing CIFAR 10
    pTransformTrain = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    pTransformTest = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    pTrainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=pTransformTrain)
    pTestSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=pTransformTest)
    pListLabelClassification = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # __draw_dataset(pTrainSet, pListLabelClassification)
    pTrainLoader = DataLoader(pTrainSet, batch_size=128, shuffle=True, num_workers=2)
    pTestLoader = DataLoader(pTestSet, batch_size=128, shuffle=False, num_workers=2)
    pModel = AlexNet().to(pDevice)
    pCriterion = torch.nn.CrossEntropyLoss()
    pOptimizer = torch.optim.SGD(pModel.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    pScheduler = torch.optim.lr_scheduler.StepLR(pOptimizer, step_size=20, gamma=0.5)
    for iEpoch in range(nEpoch):
        __process_train(iEpoch, pModel=pModel, pDataLoader=pTrainLoader, pCriterion=pCriterion, pOptimizer=pOptimizer)
        __process_evaluate(iEpoch, pModel=pModel, pDataLoader=pTestLoader, pCriterion=pCriterion, pScheduler=pScheduler)
    __process_test(pModel=pModel, pDataLoader=pTestLoader, pListLabel=pListLabelClassification)


if __name__ == "__main__":
    main()
