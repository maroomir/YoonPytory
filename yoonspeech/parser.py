import collections
import os
import yoonspeech
from os.path import splitext, basename

from tqdm import tqdm

from yoonspeech.data import YoonDataset
from yoonspeech.data import YoonObject
from yoonspeech.speech import YoonSpeech


def get_phoneme_list(strFilePath: str):
    with open(strFilePath, 'r') as pFile:
        pList = pFile.read().split('\n')[:-1]
    pList = [strTag.split(' ')[-1] for strTag in pList]
    pList = list(set(pList))
    return pList


def get_phoneme_dict(strFilePath: str):
    with open(strFilePath, 'r') as pFile:
        pList = pFile.read().split('\n')[:-1]
    pDic = {}
    for strTag in pList:
        if strTag.split(' ')[0] == 'q':
            pass
        else:
            pDic[strTag.split(' ')[0]] = strTag.split(' ')[-1]
    return pDic


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

    def get_words_in_trans(strFilePath, strID):
        with open(strFilePath) as pFile:
            pListLine = pFile.read().lower().split('\n')[:-1]
        for strLine in pListLine:
            if strID in strLine:
                strLine = strLine.replace(strID + ' ', "")
                return strLine

    def make_speech_buffer(strFile):
        return YoonSpeech(strFileName=strFile, nSamplingRate=nSamplingRate, nContextSize=nContextSize,
                          nFFTCount=nFFTCount, nMelOrder=nMelOrder, nMFCCOrder=nMFCCOrder,
                          dWindowLength=dWindowLength, dShiftLength=dShiftLength)

    pDicFeatureFile = collections.defaultdict(list)
    pDicTransFile = collections.defaultdict(dict)
    pListTrainFile = []
    pListTestFile = []
    # Extract file names
    for strRoot, strDir, pListFileName in tqdm(os.walk(strRootDir)):
        iCount = 0
        for strFileName in pListFileName:
            if splitext(strFileName)[1] == strFileType:
                strID = splitext(strFileName)[0].split('-')[0]
                pDicFeatureFile[strID].append(os.path.join(strRoot, strFileName))
                iCount += 1
                if iCount > nCountSample:
                    break
            elif splitext(strFileName)[1] == ".txt":  # Recognition the words
                strID, strPart = splitext(strFileName)[0].split('-')
                strPart = strPart.replace(".trans", "")
                pDicTransFile[strID][strPart] = os.path.join(strRoot, strFileName)
    # Listing test and train dataset
    for i, pListFileName in pDicFeatureFile.items():
        pListTrainFile.extend(pListFileName[:int(len(pListFileName) * dRatioTrain)])
        pListTestFile.extend(pListFileName[int(len(pListFileName) * dRatioTrain):])
    # Labeling speakers for Speaker recognition
    pDicSpeaker = {}
    pListSpeakers = list(pDicFeatureFile.keys())
    nSpeakersCount = len(pListSpeakers)
    for i in range(nSpeakersCount):
        pDicSpeaker[pListSpeakers[i]] = i
    # Transform data dictionary
    pDataTrain = YoonDataset(strType=strFeatureType, nSpeakers=nSpeakersCount, nPhonemes=yoonspeech.DEFAULT_PHONEME_COUNT)
    pDataTest = YoonDataset(strType=strFeatureType, nSpeakers=nSpeakersCount, nPhonemes=yoonspeech.DEFAULT_PHONEME_COUNT)
    for strFileName in pListTrainFile:
        strBase = splitext(basename(strFileName))[0]
        strID, strPart = strBase.split('-')[0], strBase.split('-')[1]
        strWord = get_words_in_trans(pDicTransFile[strID][strPart], strBase)
        pSpeech = make_speech_buffer(strFileName)
        pObject = YoonObject(nID=int(pDicSpeaker[strID]), strName=strID, pSpeech=pSpeech, strWord=strWord,
                             strType=strFeatureType)
        pDataTrain.append(pObject)
    for strFileName in pListTestFile:
        strBase = splitext(basename(strFileName))[0]
        strID, strPart = strBase.split('-')[0], strBase.split('-')[1]
        strWord = get_words_in_trans(pDicTransFile[strID][strPart], strBase)
        pSpeech = make_speech_buffer(strFileName)
        pObject = YoonObject(nID=int(pDicSpeaker[strID]), strName=strID, pSpeech=pSpeech, strWord=strWord,
                             strType=strFeatureType)
        pDataTest.append(pObject)
    print("Length of Train = {}".format(pDataTrain.__len__()))
    print("Length of Test = {}".format(pDataTest.__len__()))
    return pDataTrain, pDataTest


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
    pDataTest = YoonDataset(strType=strFeatureType, nSpeakers=nSpeakersCount, nPhonemes=yoonspeech.DEFAULT_PHONEME_COUNT)
    for strFileName in pListTestFile:
        strID = splitext(basename(strFileName))[0].split('-')[0]
        pSpeech = YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate,
                             nContextSize=nContextSize,
                             nFFTCount=nFFTCount, nMelOrder=nMelOrder, nMFCCOrder=nMFCCOrder,
                             dWindowLength=dWindowLength, dShiftLength=dShiftLength)
        pObject = YoonObject(nID=int(pDicLabel[strID]), strName=strID, pSpeech=pSpeech,
                             strType=strFeatureType)
        pDataTest.append(pObject)
    return pDataTest
