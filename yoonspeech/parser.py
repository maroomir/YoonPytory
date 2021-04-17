from yoonspeech.speech import YoonSpeech
import collections
import os
from tqdm import tqdm
import numpy
from os.path import splitext, basename


class LibriSpeechParser:
    trainDataDict: dict = {}
    testDataDict: dict = {}

    def __init__(self,
                 strRootDir: str,
                 strFileType: str = '.flac',
                 dRatioTrain=0.8):
        pDicFile = collections.defaultdict(list)
        pDicLabel = collections.defaultdict(int)
        pListTrain = []
        pListTest = []
        # Extract file names
        iCount = 0
        for strRoot, strDir, pListFileName in tqdm(os.walk(strRootDir)):
            for strFileName in pListFileName:
                if splitext(strFileName)[1] == strFileType:
                    nID = splitext(strFileName)[0].split('-')[0]
                    pDicFile[nID].append(os.path.join(strRoot, strFileName))
                    iCount += 1
        # Construct label dictionary
        pListFile = list(set(pDicFile.keys()))
        for i in range(len(pListFile)):
            pDicLabel[pListFile[i]] = i
        # Listing test and train dataset
        for i, pListFileName in pDicFile.items():
            pListTrain.extend(pListFileName[:int(len(pListFileName) * dRatioTrain)])
            pListTest.extend(pListFileName[int(len(pListFileName) * dRatioTrain):])
        # Transform data dictionary
        for strFileName in pListTrain:
            nID = splitext(basename(strFileName))[0].split('-')[0]
            self.trainDataDict[pDicLabel[nID]] = YoonSpeech(strFileName)
        for strFileName in pListTest:
            nID = splitext(basename(strFileName))[0].split('-')[0]
            self.testDataDict[pDicLabel[nID]] = YoonSpeech(strFileName)

    def to_list(self, strType: str):
        pDic = {"train_label": list(set(self.trainDataDict.keys())),
                "train_data": YoonSpeech.listing(list(set(self.trainDataDict.items())), "frame"),
                "test_label": list(set(self.testDataDict.keys())),
                "test_data": YoonSpeech.listing(list(set(self.trainDataDict.items())), "frame")
                }
        return pDic[strType]

    def to_array(self, strType: str):
        return numpy.array(self.to_list(strType))

    def to_train_dataset(self):
        pArrayLabel = self.to_array("train_label")
        pArrayData = self.to_array("train_data")
        return numpy.array(list(zip(pArrayLabel, pArrayData)))

    def to_test_dataset(self):
        pArrayLabel = self.to_array("test_label")
        pArrayData = self.to_array("train_data")
        return numpy.array(list(zip(pArrayLabel, pArrayData)))
