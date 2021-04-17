import os
import collections
import numpy
import sklearn.mixture
from tqdm import tqdm
from os.path import splitext, basename
from yoonspeech.speech import YoonSpeech


class LibriSpeechParser:
    __trainLabels: list = []
    __trainData: list = []
    __testLabels: list = []
    __testData: list = []

    def __str__(self):
        return "TRAIN COUNT : {0}, TEST COUNT : {1}".format(len(self.__trainLabels), len(self.__testLabels))

    def __init__(self,
                 strRootDir: str,
                 strFileType: str = '.flac',
                 nSamplingRate: int = 16000,
                 nCountSample: int = 100,
                 dWindowLength: float = 0.025,
                 dShiftLength: float = 0.01,
                 dRatioTrain=0.8):
        pDicFile = collections.defaultdict(list)
        pDicLabel = collections.defaultdict(int)
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
        # Construct label dictionary
        pListFile = list(set(pDicFile.keys()))
        for i in range(len(pListFile)):
            pDicLabel[pListFile[i]] = i
        # Listing test and train dataset
        for i, pListFileName in pDicFile.items():
            pListTrain.extend(pListFileName[:int(len(pListFileName) * dRatioTrain)])
            pListTest.extend(pListFileName[int(len(pListFileName) * dRatioTrain):])
        # Transform data dictionary
        # Scaling(-0.9999, 0.9999) : To protect overload error in float range
        for strFileName in pListTrain:
            nID = splitext(basename(strFileName))[0].split('-')[0]
            self.__trainLabels.append(pDicLabel[nID])
            self.__trainData.append(YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate).
                                    scaling(-0.9999, 0.9999).get_mfcc(dWindowLength=dWindowLength, dShiftLength=dShiftLength))
        for strFileName in pListTest:
            nID = splitext(basename(strFileName))[0].split('-')[0]
            self.__testLabels.append(pDicLabel[nID])
            self.__testData.append(YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate).
                                   scaling(-0.9999, 0.9999).get_mfcc(dWindowLength=dWindowLength, dShiftLength=dShiftLength))

    def to_train_dataset(self):
        pArrayLabel = numpy.array(self.__trainLabels)
        pArrayData = numpy.array(self.__trainData)
        return numpy.array(list(zip(pArrayLabel, pArrayData)))

    def to_test_dataset(self):
        pArrayLabel = numpy.array(self.__testLabels)
        pArrayData = numpy.array(self.__testData)
        return numpy.array(list(zip(pArrayLabel, pArrayData)))


# Gaussian Mixture Modeling
def train(pParser: LibriSpeechParser):
    # Make dataset
    pTrainSet = pParser.to_train_dataset()
    pTestSet = pParser.to_test_dataset()
    # Shuffle dataset
    numpy.random.shuffle(pTrainSet)
    numpy.random.shuffle(pTestSet)
    # GMM training
    nMixture = 4
    dicGMM = {}
    for i in range(len(pTrainSet)):
        dicGMM[pTrainSet[i][0]] = sklearn.mixture.GaussianMixture(n_components=nMixture, random_state=48,
                                                                  covariance_type='diag')
    for i in tqdm(range(len(pTrainSet))):
        dicGMM[pTrainSet[i][0]].fit(pTrainSet[i][1])
    # GMM test
    iAccuracy = 0
    for i, (nLabel, pData) in enumerate(pTestSet):
        dicCandidateTarget = {}
        # Calculate likelihood scores for all the trained GMMs.
        for jSpeaker in dicGMM.keys():
            dicCandidateTarget[jSpeaker] = dicGMM[jSpeaker].score(pData)
        nLabelEstimated = max(dicCandidateTarget.keys(), key=(lambda key: dicCandidateTarget[key]))
        print("Estimated: {0}, True: {1}".format(nLabelEstimated, nLabel), end='    ')
        if nLabel == nLabelEstimated:
            print("Correct!")
            iAccuracy += 1
        else:
            print("Incorrect...")
    print("Accuracy: {:.2f}".format(iAccuracy / len(pTestSet) * 100.0))
