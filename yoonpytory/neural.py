from yoonpytory.math import *


class YoonNeuron:
    inputArray: numpy.ndarray
    weightArray: numpy.ndarray
    outputArray: numpy.ndarray

    def load_source(self, strFileName: str):
        pArrayData = numpy.load(strFileName)
        self.inputArray = pArrayData['input']
        self.outputArray = pArrayData['output']

    def save_result(self, strFileName: str):
        numpy.savez(strFileName, input=self.inputArray, output=self.outputArray)

    def load_weight(self, strFileName: str):
        pArrayData = numpy.load(strFileName)
        self.weightArray = pArrayData['weight']

    def save_weight(self, strFileName: str):
        numpy.savez(strFileName, weight=self.weightArray)

    def train(self, nCountEpoch=1000, bInitWeight=True, bRunTest=True):
        nCountData, nDimensionInput = self.inputArray.shape
        self.inputArray = numpy.column_stack((self.inputArray, numpy.ones([nCountData, 1])))
        pArrayTarget = self.outputArray
        pArrayLoss = numpy.zeros([nCountEpoch, 1])
        # Weight is random value in -0.1 ~ 0.1 for training
        pArrayWeight: numpy.ndarray
        if bInitWeight:
            pArrayWeight = 0.1 * (2 * numpy.random.random((1, nDimensionInput + 1)) - 1)
        else:
            pArrayWeight = self.weightArray
        for iEpoch in range(nCountEpoch):
            pArrayWeight = gradient_descent(self.inputArray, pArrayWeight, pArrayTarget)
            pArrayOutputEstimated = logistic_regression(self.inputArray, pArrayWeight)
            pArrayLoss[iEpoch] = 1 / 2 * numpy.matmul((self.outputArray - pArrayOutputEstimated).transpose(),
                                                      (self.outputArray - pArrayOutputEstimated)) / nCountData
            if iEpoch % 100 == 0:
                print("epoch={0}, Error={1:.5f}%".format(iEpoch, float(pArrayLoss[iEpoch]) * 100))
        self.weightArray = pArrayWeight.copy()
        # Test weight
        if bRunTest:
            self.process(False)

    def process(self, bSaveOutput=False):
        nCountData, nDimensionInput = self.inputArray.shape
        pArrayResult = (logistic_regression(self.inputArray, self.weightArray) >= 0.5) * 1
        print('The accuracy is {:.5f}%'.format(numpy.sum(self.outputArray == pArrayResult) / nCountData * 100))
        if bSaveOutput:
            self.outputArray = pArrayResult
        return pArrayResult
