import collections
import os
from os.path import splitext, basename

import numpy
from tqdm import tqdm

from yoonspeech.speech import YoonSpeech


class YoonParser:
    _trainLabels: list = []
    _trainData: list = []
    _testLabels: list = []
    _testData: list = []

    def __str__(self):
        return "TRAIN COUNT : {0}, TEST COUNT : {1}".format(len(self._trainLabels), len(self._testLabels))

    def to_train_dataset(self):
        pArrayLabel = numpy.array(self._trainLabels)
        pArrayData = numpy.array(self._trainData)
        return numpy.array(list(zip(pArrayLabel, pArrayData)))

    def to_test_dataset(self):
        pArrayLabel = numpy.array(self._testLabels)
        pArrayData = numpy.array(self._testData)
        return numpy.array(list(zip(pArrayLabel, pArrayData)))


class LibriSpeechParser(YoonParser):
    def __init__(self,
                 strRootDir: str,
                 strFileType: str = '.flac',
                 nSamplingRate: int = 16000,
                 nCountSample: int = 1000,
                 dWindowLength: float = 0.025,
                 dShiftLength: float = 0.01,
                 dRatioTrain=0.8):
        pDicFile = collections.defaultdict(list)
        pListTrain = []
        pListTest = []
        # Extract file names
        for strRoot, strDir, pListFileName in tqdm(os.walk(strRootDir)):
            iCount = 0
            for strFileName in pListFileName:
                if splitext(strFileName)[1] == strFileType:
                    nID = splitext(strFileName)[0].split('-')[0]
                    pDicFile[nID].append(os.path.join(strRoot, strFileName))
                    iCount += 1
                    if iCount > nCountSample:
                        break
        # Listing test and train dataset
        for i, pListFileName in pDicFile.items():
            pListTrain.extend(pListFileName[:int(len(pListFileName) * dRatioTrain)])
            pListTest.extend(pListFileName[int(len(pListFileName) * dRatioTrain):])
        # Transform data dictionary
        # Scaling(-0.9999, 0.9999) : To protect overload error in float range
        for strFileName in pListTrain:
            nID = splitext(basename(strFileName))[0].split('-')[0]
            self._trainLabels.append(nID)
            self._trainData.append(YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate,
                                              dWindowLength=dWindowLength, dShiftLength=dShiftLength).
                                   scaling(-0.9999, 0.9999).get_mfcc())
        for strFileName in pListTest:
            nID = splitext(basename(strFileName))[0].split('-')[0]
            self._testLabels.append(nID)
            self._testData.append(YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate,
                                             dWindowLength=dWindowLength, dShiftLength=dShiftLength).
                                  scaling(-0.9999, 0.9999).get_mfcc())
