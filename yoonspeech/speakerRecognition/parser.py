import collections
import os
from os.path import splitext, basename

import numpy
from tqdm import tqdm

from yoonspeech.speech import YoonSpeech
from yoonspeech.data import YoonObject
from yoonspeech.data import YoonDataset


def parse_librispeech_trainer(strRootDir: str,
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
    pDicFile = collections.defaultdict(list)
    pListTrainFile = []
    pListTestFile = []
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
        pListTrainFile.extend(pListFileName[:int(len(pListFileName) * dRatioTrain)])
        pListTestFile.extend(pListFileName[int(len(pListFileName) * dRatioTrain):])
    # Labeling speakers for PyTorch Training
    pDicLabel = {}
    pListSpeakers = list(pDicFile.keys())
    nSpeakersCount = len(pListSpeakers)
    for i in range(nSpeakersCount):
        pDicLabel[pListSpeakers[i]] = i
    # Transform data dictionary
    pDataTrain = YoonDataset(strType=strFeatureType, nCount=nSpeakersCount)
    pDataTest = YoonDataset(strType=strFeatureType, nCount=nSpeakersCount)
    for strFileName in pListTrainFile:
        strID = splitext(basename(strFileName))[0].split('-')[0]
        pSpeech = YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate,
                             nContextSize=nContextSize,
                             nFFTCount=nFFTCount, nMelOrder=nMelOrder, nMFCCOrder=nMFCCOrder,
                             dWindowLength=dWindowLength, dShiftLength=dShiftLength)
        pObject = YoonObject(nID=int(pDicLabel[strID]), strName=strID, pSpeech=pSpeech, strType=strFeatureType)
        pDataTrain.append(pObject)
    for strFileName in pListTestFile:
        strID = splitext(basename(strFileName))[0].split('-')[0]
        pSpeech = YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate,
                             nContextSize=nContextSize,
                             nFFTCount=nFFTCount, nMelOrder=nMelOrder, nMFCCOrder=nMFCCOrder,
                             dWindowLength=dWindowLength, dShiftLength=dShiftLength)
        pObject = YoonObject(nID=int(pDicLabel[strID]), strName=strID, pSpeech=pSpeech, strType=strFeatureType)
        pDataTest.append(pObject)
    return pDataTrain.__copy__(), pDataTest.__copy__()


def parse_librispeech_tester(strRootDir: str,
                             strFileType: str = '.flac',
                             nCountSample: int = 1000,
                             nSamplingRate: int = 16000,
                             nFFTCount: int = 512,
                             nMelOrder: int = 24,
                             nMFCCOrder: int = 13,
                             nContextSize: int = 10,
                             dWindowLength: float = 0.025,
                             dShiftLength: float = 0.01,
                             strFeatureType: str = "mfcc"):
    pDicFile = collections.defaultdict(list)
    pListTestFile = []
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
        pListTestFile.extend(pListFileName[:int(len(pListFileName))])
    # Labeling speakers for PyTorch Training
    pDicLabel = {}
    pListSpeakers = list(pDicFile.keys())
    nSpeakersCount = len(pListSpeakers)
    for i in range(nSpeakersCount):
        pDicLabel[pListSpeakers[i]] = i
    # Transform data dictionary
    pDataTest = YoonDataset(strType=strFeatureType, nCount=nSpeakersCount)
    for strFileName in pListTestFile:
        strID = splitext(basename(strFileName))[0].split('-')[0]
        pSpeech = YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate,
                             nContextSize=nContextSize,
                             nFFTCount=nFFTCount, nMelOrder=nMelOrder, nMFCCOrder=nMFCCOrder,
                             dWindowLength=dWindowLength, dShiftLength=dShiftLength)
        pObject = YoonObject(nID=int(pDicLabel[strID]), strName=strID, pSpeech=pSpeech, strType=strFeatureType)
        pDataTest.append(pObject)
    return pDataTest.__copy__()
