from yoonspeech.parser import LibriSpeechParser
import numpy
import sklearn
import tqdm


# Gaussian Mixture Modeling
def train(pParser: LibriSpeechParser):
    # Make dataset
    pTrainDataSet = pParser.to_train_dataset()
    pTestDataSet = pParser.to_test_dataset()
    # Shuffle dataset
    numpy.random.shuffle(pTrainDataSet)
    numpy.random.shuffle(pTestDataSet)
    # GMM training
    nMixture = 4
    dicGMM = {}
    for i in range(len(pTrainDataSet)):
        dicGMM[pTrainDataSet[i][0]] = sklearn.mixture.GaussianMixture(n_components=nMixture, random_state=48,
                                                                      covariance_type='diag')
    for i in tqdm(range(len(pTrainDataSet))):
        dicGMM[pTestDataSet[i][0]].fit(pTrainDataSet[i][1])
    # GMM test
    iAccuracy = 0
    for i, (nLabel, pData) in enumerate(pTestDataSet):
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
    print("Accuracy: {.2f}".format(iAccuracy/len(pTestDataSet)*100.0))