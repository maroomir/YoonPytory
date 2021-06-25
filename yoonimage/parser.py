import os
import pickle
import numpy

from yoonimage.data import YoonDataset, YoonObject
from yoonimage.image import YoonImage


def parse_cifar10_trainer(strRootDir: str,
                          dRatioTrain: float = 0.8,
                          strMode: str = "alexnet"  # alexnet, resnet, vgg
                          ):
    # Read the label names
    strLabelFile = os.path.join(strRootDir, "batches.meta")
    with open(strLabelFile, 'rb') as pFile:
        pDicLabel = pickle.load(pFile)
        pArrayName = pDicLabel['label_names']
    # Read the data
    pListTrainFile = [os.path.join(strRootDir, "data_batch_{}".format(i + 1)) for i in range(5)]
    pArrayData = []
    pArrayLabel = []
    for strPath in pListTrainFile:
        try:
            with open(strPath, 'rb') as pFile:
                pDicData = pickle.load(pFile, encoding='bytes')
            pArrayData.append(pDicData['data'])
            pArrayLabel.append(pDicData['label'])
        except:
            FileExistsError("{} file is not exists".format(strPath))
    pArrayData = numpy.concatenate(pArrayData, axis=0)
    pArrayLabel = numpy.concatenate(pArrayLabel, axis=0)
    # Transform data array to YoonDataset
    if pArrayData.shape[0] != pArrayLabel.shape[0]:
        ValueError("The label and data size is not equal")
    pDataTrain = YoonDataset()
    pDataEval = YoonDataset()
    nCutLine = int(pArrayData.shape[0] * dRatioTrain)
    for i in range(nCutLine):
        pImage = YoonImage.from_array(pArrayData[i], nWidth=32, nHeight=32, nChannel=3)
        nLabel = pArrayLabel[i]
        pObject = YoonObject(nID=nLabel, strName=pArrayName[nLabel], pImage=pImage)
        pDataTrain.append(pObject)
    for i in range(nCutLine, pArrayData.shape[0]):
        pImage = YoonImage.from_array(pArrayData[i], nWidth=32, nHeight=32, nChannel=3)
        nLabel = pArrayLabel[i]
        pObject = YoonObject(nID=nLabel, strName=pArrayName[nLabel], pImage=pImage)
        pDataEval.append(pObject)
    print("Length of Train = {}".format(pDataTrain.__len__()))
    print("Length of Test = {}".format(pDataEval.__len__()))
    nDimOutput = pArrayName.shape[0]  # 10 (CIFAR-10)
    return nDimOutput, pDataTrain, pDataEval


def parse_cifar10_tester(strRootDir: str,
                         strMode: str = "alexnet"  # alexnet, resnet, unet, vgg
                         ):
    # Read the label names
    strLabelFile = os.path.join(strRootDir, "batches.meta")
    with open(strLabelFile, 'rb') as pFile:
        pDicLabel = pickle.load(pFile)
        pArrayName = pDicLabel['label_names']
    # Read the data
    strDataFile = os.path.join(strRootDir, "test_batch")
    with open(strDataFile, 'rb') as pFile:
        pDicData = pickle.load(pFile, encoding='bytes')
    pArrayData = pDicData['data']
    pArrayLabel = pDicData['label']
    # Transform data array to YoonDataset
    if pArrayData.shape[0] != pArrayLabel.shape[0]:
        ValueError("The label and data size is not equal")
    pDataTest = YoonDataset()
    for i in range(pArrayData.shape[0]):
        pImage = YoonImage.from_array(pArrayData[i], nWidth=32, nHeight=32, nChannel=3)
        nLabel = pArrayLabel[i]
        pObject = YoonObject(nID=nLabel, strName=pArrayName[nLabel], pImage=pImage)
        pDataTest.append(pObject)
    print("Length of Test = {}".format(pDataTest.__len__()))
    nDimOutput = pArrayName.shape[0]  # 10 (CIFAR-10)
    return nDimOutput, pDataTest
