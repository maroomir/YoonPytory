import numpy
import pickle
import sklearn.mixture
from tqdm import tqdm
from yoonspeech.parser import YoonParser
from yoonspeech.speech import YoonSpeech


# Gaussian Mixture Modeling
def gmm_train(pParser: YoonParser, strModelPath: str):
    # Make dataset
    pTrainSet = pParser.to_train_dataset()
    pTestSet = pParser.to_test_dataset()
    # Shuffle dataset
    numpy.random.shuffle(pTrainSet)
    numpy.random.shuffle(pTestSet)
    # GMM training
    nMixture = 4
    pDicGMM = {}
    for i in range(len(pTrainSet)):
        pDicGMM[pTrainSet[i][0]] = sklearn.mixture.GaussianMixture(n_components=nMixture, random_state=48,
                                                                   covariance_type='diag')
    for i in tqdm(range(len(pTrainSet))):
        pDicGMM[pTrainSet[i][0]].fit(pTrainSet[i][1])
    # GMM test
    iAccuracy = 0
    for i, (nLabel, pData) in enumerate(pTestSet):
        pDicCandidateScore = {}
        # Calculate likelihood scores for all the trained GMMs.
        for jSpeaker in pDicGMM.keys():
            pDicCandidateScore[jSpeaker] = pDicGMM[jSpeaker].score(pData)
        nLabelEstimated = max(pDicCandidateScore.keys(), key=(lambda key: pDicCandidateScore[key]))
        print("Estimated: {0}, Score: {1:.2f}, True: {2}".
              format(nLabelEstimated, pDicCandidateScore[nLabelEstimated], nLabel), end='    ')
        if nLabel == nLabelEstimated:
            print("Correct!")
            iAccuracy += 1
        else:
            print("Incorrect...")
    print("Accuracy: {:.2f}".format(iAccuracy / len(pTestSet) * 100.0))
    # Save GMM modeling
    with open(strModelPath, 'wb') as pFile:
        pickle.dump(pDicGMM, pFile)
        print("Save {} GMM models".format(len(pDicGMM)))


def gmm_parser_recognition(pParser: YoonParser, strModelPath: str):
    # Load GMM modeling
    with open(strModelPath, 'rb') as pFile:
        pDicGMM = pickle.load(pFile)
    pTestSet = pParser.to_test_dataset()
    # GMM test
    iAccuracy = 0
    for i, (nLabel, pData) in enumerate(pTestSet):
        pDicCandidateScore = {}
        # Calculate likelihood scores for all the trained GMMs.
        for jSpeaker in pDicGMM.keys():
            pDicCandidateScore[jSpeaker] = pDicGMM[jSpeaker].score(pData)
        nLabelEstimated = max(pDicCandidateScore.keys(), key=(lambda key: pDicCandidateScore[key]))
        print("Estimated: {0}, Score: {1:.2f}, True: {2}".
              format(nLabelEstimated, pDicCandidateScore[nLabelEstimated], nLabel), end='    ')
        if nLabel == nLabelEstimated:
            print("Correct!")
            iAccuracy += 1
        else:
            print("Incorrect...")
    print("Accuracy: {:.2f}".format(iAccuracy / len(pTestSet) * 100.0))


def gmm_speech_recognition(pSpeech: YoonSpeech, strModelPath: str):
    # Load GMM modeling
    with open(strModelPath, 'rb') as pFile:
        pDicGMM = pickle.load(pFile)
    pTestData = pSpeech.scaling(-0.9999, 0.9999).get_mfcc()
    pDicTestScore = {}
    for iSpeaker in pDicGMM.keys():
        pDicTestScore[iSpeaker] = pDicGMM[iSpeaker].score(pTestData)
    nLabelEsimated = max(pDicTestScore.keys(), key=(lambda key: pDicTestScore[key]))
    print("Estimated: {0}, Score : {1:.2f}".format(nLabelEsimated, pDicTestScore[nLabelEsimated]))
    return nLabelEsimated
