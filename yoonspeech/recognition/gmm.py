import numpy
import pickle
import sklearn.mixture
from tqdm import tqdm
from yoonspeech.speech import YoonSpeech
from yoonspeech.data import YoonDataset


# Gaussian Mixture Modeling
def train(pTrainData: YoonDataset, pTestData: YoonDataset, strModelPath: str):
    # Make dataset
    pTrainSet = pTrainData.to_gmm_dataset()
    pTestSet = pTestData.to_gmm_dataset()
    # Shuffle dataset
    numpy.random.shuffle(pTrainSet)
    numpy.random.shuffle(pTestSet)
    # GMM training
    nMixture = 4
    pDicGMM = {}
    for i in range(len(pTrainSet)):
        pDicGMM[pTrainSet[i][1]] = sklearn.mixture.GaussianMixture(n_components=nMixture, random_state=48,
                                                                   covariance_type='diag')
        # The covariance type is "full" matrix if we use "Deltas" data
    for i in tqdm(range(len(pTrainSet))):
        pDicGMM[pTrainSet[i][1]].fit(pTrainSet[i][0])
    # GMM test
    iAccuracy = 0
    for i, (pInputData, nTargetLabel) in enumerate(pTestSet):
        pDicCandidateScore = {}
        # Calculate likelihood scores for all the trained GMMs.
        for jSpeaker in pDicGMM.keys():
            pDicCandidateScore[jSpeaker] = pDicGMM[jSpeaker].score(pInputData)
        nLabelEstimated = max(pDicCandidateScore.keys(), key=(lambda key: pDicCandidateScore[key]))
        print("Estimated: {0}, Score: {1:.2f}, True: {2}".
              format(nLabelEstimated, pDicCandidateScore[nLabelEstimated], nTargetLabel), end='    ')
        if nTargetLabel == nLabelEstimated:
            print("Correct!")
            iAccuracy += 1
        else:
            print("Incorrect...")
    print("Accuracy: {:.2f}".format(iAccuracy / len(pTestSet) * 100.0))
    # Save GMM modeling
    with open(strModelPath, 'wb') as pFile:
        pickle.dump(pDicGMM, pFile)
        print("Save {} GMM models".format(len(pDicGMM)))


def test(pTestData: YoonDataset, strModelPath: str):
    # Load GMM modeling
    with open(strModelPath, 'rb') as pFile:
        pDicGMM = pickle.load(pFile)
    pTestSet = pTestData.to_gmm_dataset()
    # GMM test
    iAccuracy = 0
    for i, (pInputData, nTargetLabel) in enumerate(pTestSet):
        pDicCandidateScore = {}
        # Calculate likelihood scores for all the trained GMMs.
        for jSpeaker in pDicGMM.keys():
            pDicCandidateScore[jSpeaker] = pDicGMM[jSpeaker].score(pInputData)
        nLabelEstimated = max(pDicCandidateScore.keys(), key=(lambda key: pDicCandidateScore[key]))
        print("Estimated: {0}, Score: {1:.2f}, True: {2}".
              format(nLabelEstimated, pDicCandidateScore[nLabelEstimated], nTargetLabel), end='    ')
        if nTargetLabel == nLabelEstimated:
            print("Correct!")
            iAccuracy += 1
        else:
            print("Incorrect...")
    print("Accuracy: {:.2f}".format(iAccuracy / len(pTestSet) * 100.0))


def recognition(pSpeech: YoonSpeech, strModelPath: str, strFeatureType="mfcc"):
    # Load GMM modeling
    with open(strModelPath, 'rb') as pFile:
        pDicGMM = pickle.load(pFile)
    if strFeatureType == "mel":
        pTestBuffer = pSpeech.scaling(-0.9999, 0.9999).get_log_mel_spectrum()
    elif strFeatureType == "mfcc":
        pTestBuffer = pSpeech.scaling(-0.9999, 0.9999).get_mfcc()
    else:
        Exception("Feature type is not correct")
    pDicTestScore = {}
    for iSpeaker in pDicGMM.keys():
        pDicTestScore[iSpeaker] = pDicGMM[iSpeaker].score(pTestBuffer)
    nLabelEstimated = max(pDicTestScore.keys(), key=(lambda key: pDicTestScore[key]))
    print("Estimated: {0}, Score : {1:.2f}".format(nLabelEstimated, pDicTestScore[nLabelEstimated]))
    return nLabelEstimated
