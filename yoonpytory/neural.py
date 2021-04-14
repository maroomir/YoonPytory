import numpy
import matplotlib.pyplot


def _sigmoid(pX: numpy.ndarray):
    return numpy.exp(-numpy.logaddexp(0, -pX))


def _logistic_regression(pX: numpy.ndarray, pW: numpy.ndarray):  # Output : Calculate output (pY)
    # X is input (row-wise sample)
    # W is weight
    pArrayV = numpy.matmul(pW, pX.transpose())
    pArrayV = pArrayV.transpose()
    pArrayY = _sigmoid(pArrayV)
    return pArrayY


def _gradient_descent(pX: numpy.ndarray, pW: numpy.ndarray, pY: numpy.ndarray,
                      dAlpha=0.001):  # Output : improved weight (W+1)
    # X is input
    nCountX, nDimensionX = pX.shape
    # W is weight estimated (not constructed)
    nCountW, nDimensionW = pW.shape
    # Y is target output (Right answer)
    nCountY, nDimensionY = pY.shape
    if nCountW != 1 and nDimensionY != 1:
        raise Exception("The layer count is not only one")
    if nCountX != nCountY:
        raise Exception("Array X count {0}, Delta count {1} is not equal".format(nCountX, nCountY))
    if nDimensionX != nDimensionW:
        raise Exception("X Dimension {0}, W Dimension {1} is not equal".format(nDimensionX, nDimensionW))
    for i in range(nCountX):
        iArrayX = pX[i, :]  # iInput length = nDimension
        dTarget = pY[i]  # Size = 1
        # feedforward process
        dY = _logistic_regression(iArrayX, pW)
        # backward process
        dError = dTarget - dY
        iGradient = dAlpha * dError * iArrayX
        pW = pW + iGradient.transpose()
    return pW


def _back_propagation(pX: numpy.ndarray, pListW: list, pY: numpy.ndarray,
                      dAlpha=0.01):  # Output : improved weight List(W+1)
    # X is input
    # W is weight estimated (not constructed)
    # Y is target output (Right answer)
    nCountX, nDimensionX = pX.shape
    for i in range(nCountX):
        iArrayX = numpy.array(pX[i, :], ndmin=2)
        dTarget = numpy.array(pY[i], ndmin=2)
        # feedforward process
        pListY = []
        iLayer = iArrayX.copy()
        for j in range(0, len(pListW), 1):
            assert isinstance(pListW[j], numpy.ndarray)
            iLayer = _logistic_regression(iLayer.copy(), pListW[j])
            pListY.append(iLayer.copy())
        # backward process
        pListError = [numpy.ndarray] * len(pListW)
        pListDelta = [numpy.ndarray] * len(pListW)
        for j in range(len(pListW) - 1, -1, -1):
            if j == len(pListW) - 1:
                pListError[j] = dTarget - pListY[j]
                pListDelta[j] = pListError[j]
            else:
                pListError[j] = numpy.matmul(pListW[j + 1].transpose(), pListDelta[j + 1])
                pListDelta[j] = pListY[j].transpose() * (1 - pListY[j].transpose()) * pListError[j]
        # update weight
        for j in range(0, len(pListW), 1):
            if j == 0:
                iGradient = dAlpha * numpy.matmul(pListDelta[j], iArrayX)
            else:
                iGradient = dAlpha * numpy.matmul(pListDelta[j], pListY[j - 1])
            pListW[j] = pListW[j] + iGradient.copy()
    return pListW


class YoonNeuron:
    input: numpy.ndarray
    weight: numpy.ndarray
    output: numpy.ndarray

    def load_source(self, strFileName: str):
        pArrayData = numpy.load(strFileName)
        self.input = pArrayData['input']
        self.output = pArrayData['output']

    def save_result(self, strFileName: str):
        numpy.savez(strFileName, input=self.input, output=self.output)

    def load_weight(self, strFileName: str):
        pArrayData = numpy.load(strFileName)
        self.weight = pArrayData['weight']

    def save_weight(self, strFileName: str):
        numpy.savez(strFileName, weight=self.weight)

    def train(self, nCountEpoch=1000, dScale=0.1, bInitWeight=True, bRunTest=True):
        nCountData, nDimInput = self.input.shape
        pInputTransform = numpy.column_stack((self.input, numpy.ones([nCountData, 1])))
        pArrayTarget = self.output
        pArrayLoss = numpy.zeros([nCountEpoch, 1])
        pArrayWeight: numpy.ndarray
        if bInitWeight:
            # Weight is random value in -0.1 ~ 0.1 for training
            pArrayWeight = dScale * (2 * numpy.random.random((1, nDimInput + 1)) - 1)
        else:
            pArrayWeight = self.weight.copy()
        # Train
        for iEpoch in range(nCountEpoch):
            pArrayWeight = _gradient_descent(pX=pInputTransform, pW=pArrayWeight, pY=pArrayTarget)
            pArrayOutputEstimated = _logistic_regression(pInputTransform, pArrayWeight)
            pArrayLoss[iEpoch] = 1 / 2 * numpy.matmul((self.output - pArrayOutputEstimated).transpose(),
                                                      (self.output - pArrayOutputEstimated)) / nCountData
            if iEpoch % 100 == 0:
                print("epoch={0}, Error={1:.5f}%".format(iEpoch, float(pArrayLoss[iEpoch]) * 100))
        self.weight = pArrayWeight.copy()
        # Test weight
        if bRunTest:
            self.process(False)

    def process(self, bSaveOutput=False):
        nCountData, nDimensionInput = self.input.shape
        pInputTransform = numpy.column_stack((self.input, numpy.ones([nCountData, 1])))
        pArrayResult = (_logistic_regression(pInputTransform, self.weight) >= 0.5) * 1
        print('The accuracy is {:.5f}%'.format(numpy.sum(self.output == pArrayResult) / nCountData * 100))
        if bSaveOutput:
            self.output = pArrayResult
        return pArrayResult

    def show_plot(self):
        posInit1 = numpy.where(self.output == 0)[0]
        posInit2 = numpy.where(self.output == 1)[0]
        tuplePosInit = (posInit1, posInit2)
        tupleColor = ("red", "green")
        for iPos, iColor in zip(tuplePosInit, tupleColor):
            matplotlib.pyplot.scatter(self.input[iPos, 0], self.input[iPos, 1], alpha=1.0, c=iColor)
        dStep = 0.025
        listX = numpy.arange(0.0, 1.0 + dStep, dStep)
        listY = numpy.arange(0.0, 1.0 + dStep, dStep)
        pXMesh, pYMesh = numpy.meshgrid(listX, listY)
        pZMesh = numpy.zeros(pXMesh.shape)
        for iX in range(pXMesh.shape[0]):
            for iY in range(pYMesh.shape[0]):
                listInput = numpy.array([pXMesh[iX][iY], pYMesh[iX][iY], 1], ndmin=2)
                pZMesh[iX][iY] = _logistic_regression(listInput, self.weight)
        matplotlib.pyplot.contour(listX, listY, pZMesh, (0.49, 0.51))
        matplotlib.pyplot.show()


class YoonNetwork:
    input: numpy.ndarray
    weights: list
    output: numpy.ndarray

    def load_source(self, strFileName: str):
        pArrayData = numpy.load(strFileName)
        self.input = pArrayData['input']
        self.output = pArrayData['output']

    def save_result(self, strFileName: str):
        numpy.savez(strFileName, input=self.input, output=self.output)

    def load_weight(self, strFileName: str):
        pArrayData = numpy.load(strFileName)
        self.weights = []
        for i in range(len(pArrayData.files)):
            self.weights.append(pArrayData['arr_{0}'.format(i)])
        print("The weight order is {0}".format(len(self.weights)))

    def save_weight(self, strFileName: str):
        pArgs = (self.weights[i] for i in range(len(self.weights)))
        numpy.savez(strFileName, *pArgs)

    def train(self, nCountEpoch=5000, nSizeLayer=10, nOrder=3, nDimOutput=1, dScale=1.0,
              bInitWeight=True, bRunTest=True):
        if nCountEpoch < 1000 or nOrder < 2 or nSizeLayer < 10:
            raise Exception("Train arguments is too little, Epoch: {0}, Size {1}, Order {2}".
                            format(nCountEpoch, nSizeLayer, nOrder))
        nCountData, nDimInput = self.input.shape
        pInputTransform = numpy.column_stack((self.input, numpy.ones([nCountData, 1])))
        pArrayTarget = self.output
        pArrayLoss = numpy.zeros([nCountEpoch, 1])
        # Weight is random value in -0.1 ~ 0.1 for training
        pListWeight: list
        if self.weights is not None and not bInitWeight and nOrder == len(self.weights):
            pListWeight = self.weights.copy()
        else:
            # Weight is random value in -0.1 ~ 0.1 for training
            pListWeight = []
            for iOrder in range(nOrder):
                if iOrder == 0:
                    pListWeight.append(dScale * (2 * numpy.random.random((nSizeLayer, nDimInput + 1)) - 1))
                elif iOrder == nOrder - 1:
                    pListWeight.append(dScale * (2 * numpy.random.random((nDimOutput, nSizeLayer)) - 1))
                else:
                    pListWeight.append(dScale * (2 * numpy.random.random((nSizeLayer, nSizeLayer)) - 1))
        # Train
        for iEpoch in range(nCountEpoch):
            pListWeight = _back_propagation(pX=pInputTransform, pListW=pListWeight, pY=pArrayTarget)
            pArrayOutputEstimated = self.__feed_forward_network(pX=pInputTransform, pListW=pListWeight)
            pArrayLoss[iEpoch] = 1 / 2 * numpy.matmul((self.output - pArrayOutputEstimated).transpose(),
                                                      (self.output - pArrayOutputEstimated)) / nCountData
            if iEpoch % 100 == 0:
                print("epoch={0}, Error={1:.5f}%".format(iEpoch, float(pArrayLoss[iEpoch]) * 100))
        self.weights = pListWeight.copy()
        # Test weight
        if bRunTest:
            self.process(False)

    def process(self, bSaveOutput=False):
        nCountData, nDimensionInput = self.input.shape
        pArrayResult = (self.__feed_forward_network() >= 0.5) * 1
        print('The accuracy is {:.5f}%'.format(numpy.sum(self.output == pArrayResult) / nCountData * 100))
        if bSaveOutput:
            self.output = pArrayResult
        return pArrayResult

    def __feed_forward_network(self, pX: list = None, pListW: list = None):
        if pX is None:
            nCountData, nDimInput = self.input.shape
            pX = numpy.column_stack((self.input, numpy.ones([nCountData, 1])))
        if pListW is None:
            pListW = self.weights
        pResult = pX
        for j in range(0, len(pListW), 1):
            assert isinstance(pListW[j], numpy.ndarray)
            pResult = _logistic_regression(pResult.copy(), pListW[j])
        return pResult

    def show_plot(self):
        posInit1 = numpy.where(self.output == 0)[0]
        posInit2 = numpy.where(self.output == 1)[0]
        tuplePosInit = (posInit1, posInit2)
        tupleColor = ("red", "green")
        for iPos, iColor in zip(tuplePosInit, tupleColor):
            matplotlib.pyplot.scatter(self.input[iPos, 0], self.input[iPos, 1], alpha=1.0, c=iColor)
        dStep = 0.025
        listX = numpy.arange(0.0, 1.0 + dStep, dStep)
        listY = numpy.arange(0.0, 1.0 + dStep, dStep)
        pXMesh, pYMesh = numpy.meshgrid(listX, listY)
        pZMesh = numpy.zeros(pXMesh.shape)
        for iX in range(pXMesh.shape[0]):
            for iY in range(pYMesh.shape[0]):
                listInput = numpy.array([pXMesh[iX][iY], pYMesh[iX][iY], 1], ndmin=2)
                pZMesh[iX][iY] = self.__feed_forward_network(listInput)
        matplotlib.pyplot.contour(listX, listY, pZMesh, (0.49, 0.51))
        matplotlib.pyplot.show()