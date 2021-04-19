import collections
import os
from os.path import splitext, basename

import numpy
from tqdm import tqdm

from yoonspeech.speech import YoonSpeech


class YoonParser:
    # Speech setting
    fftCount: int
    samplingRate: int
    windowLength: float
    shiftLength: float
    featureType: str
    melOrder: int
    mfccOrder: int
    contextSize: int
    # Data for label
    _trainLabels: list = []
    _trainNames: list = []
    _trainData: list = []
    _testLabels: list = []
    _testNames: list = []
    _testData: list = []

    def __str__(self):
        return "TRAIN COUNT : {0}, TEST COUNT : {1}".format(len(self._trainLabels), len(self._testLabels))

    def get_train_count(self):
        return len(self._trainLabels)

    def get_test_count(self):
        return len(self._testLabels)

    def get_train_label(self, i):
        return self._trainLabels[i]

    def get_train_name(self, i):
        return self._trainNames[i]

    def get_train_data(self, i):
        return self._trainData[i]

    def get_test_label(self, i):
        return self._testLabels[i]

    def get_test_name(self, i):
        return self._testNames[i]

    def get_test_data(self, i):
        return self._testData[i]

    def get_data_dimension(self):
        if self.featureType == "mfcc":
            return self.mfccOrder
        elif self.featureType == "mel":
            return self.melOrder
        elif self.featureType == "deltas":
            return self.mfccOrder * 3 * self.contextSize    # MFCC * delta * delta-delta
        else:
            Exception("Feature type is not correct")

    def to_train_dataset(self):
        pArrayNames = numpy.array(self._trainNames)
        pArrayData = numpy.array(self._trainData)
        return numpy.array(list(zip(pArrayNames, pArrayData)))

    def to_test_dataset(self):
        pArrayNames = numpy.array(self._testNames)
        pArrayData = numpy.array(self._testData)
        return numpy.array(list(zip(pArrayNames, pArrayData)))


class LibriSpeechParser(YoonParser):
    def __init__(self,
                 strRootDir: str,
                 strFileType: str = '.flac',
                 nCountSample: int = 1000,
                 nSamplingRate: int = 16000,
                 nFFTCount: int = 512,
                 nMelOrder: int = 24,
                 nMFCCOrder: int = 13,
                 nContextSize: int = 10,
                 dWindowLength: float = 0.025,
                 dShiftLength: float = 0.01,
                 strFeatureType: str = "mfcc",
                 dRatioTrain: float = 0.8):
        # Init parameter
        self.samplingRate = nSamplingRate
        self.fftCount = nFFTCount
        self.mfccOrder = nMFCCOrder
        self.melOrder = nMelOrder
        self.contextSize = nContextSize
        self.windowLength = dWindowLength
        self.shiftLength = dShiftLength
        self.featureType = strFeatureType
        pDicFile = collections.defaultdict(list)
        pListTrain = []
        pListTest = []
        # Extract file names
        for strRoot, strDir, pListFileName in tqdm(os.walk(strRootDir)):
            iCount = 0
            for strFileName in pListFileName:
                if splitext(strFileName)[1] == strFileType:
                    strID = splitext(strFileName)[0].split('-')[0]
                    pDicFile[strID].append(os.path.join(strRoot, strFileName))
                    iCount += 1
                    if iCount > nCountSample:
                        break
        # Listing test and train dataset
        for i, pListFileName in pDicFile.items():
            pListTrain.extend(pListFileName[:int(len(pListFileName) * dRatioTrain)])
            pListTest.extend(pListFileName[int(len(pListFileName) * dRatioTrain):])
        # Labeling speakers
        pListSpeakerTrain = []
        pListSpeakerTest = []
        for i in range(len(pListTrain)):
            strKey = os.path.splitext(os.path.basename(pListTrain[i]))[0].split('-')[0]
            pListSpeakerTrain.append(strKey)
        for i in range(len(pListTest)):
            strKey = os.path.splitext(os.path.basename(pListTest[i]))[0].split('-')[0]
            pListSpeakerTest.append(strKey)
        pDicLabelTrain = {}
        pDicLabelTest = {}
        for i in range(len(pListSpeakerTrain)):
            pDicLabelTrain[pListSpeakerTrain[i]] = i  # 'Speaker_No' : i
        for i in range(len(pListSpeakerTest)):
            pDicLabelTest[pListSpeakerTest[i]] = i  # 'Speaker_No' : i
        # Transform data dictionary
        # Scaling(-0.9999, 0.9999) : To protect overload error in float range
        for strFileName in pListTrain:
            strID = splitext(basename(strFileName))[0].split('-')[0]
            pSpeech = YoonSpeech(strFileName=strFileName, nSamplingRate=self.samplingRate, nContextSize=self.contextSize,
                                 nFFTCount=self.fftCount, nMelOrder=self.melOrder, nMFCCOrder=self.mfccOrder,
                                 dWindowLength=self.windowLength, dShiftLength=self.shiftLength)
            pFeature = pSpeech.get_feature(self.featureType)
            self._trainNames.append(strID)
            self._trainLabels.append(int(pDicLabelTrain[strID]))  # The ID is Integer for blocking Error in Collator
            self._trainData.append(pFeature)
        for strFileName in pListTest:
            strID = splitext(basename(strFileName))[0].split('-')[0]
            pSpeech = YoonSpeech(strFileName=strFileName, nSamplingRate=self.samplingRate, nContextSize=self.contextSize,
                                 nFFTCount=self.fftCount, nMelOrder=self.melOrder, nMFCCOrder=self.mfccOrder,
                                 dWindowLength=self.windowLength, dShiftLength=self.shiftLength)
            pFeature = pSpeech.get_feature(self.featureType)
            self._testNames.append(int(strID))
            self._testLabels.append(int(pDicLabelTest[strID]))  # The ID is Integer for blocking Error in Collator
            self._testData.append(pFeature)
