import numpy


def least_square(pX: numpy.ndarray, pY: numpy.ndarray):
    if pX.size != pY.size:
        raise Exception("Array X size:{0}, Y size:{1} is not equal".format(pX.size, pY.size))
    nCount = pX.size
    # Calculate Model per least-square
    dMeanX, dMeanY = pX.mean(), pY.mean()
    dCrossDeviationYX = numpy.sum(pY * pX) - nCount * dMeanX * dMeanY
    dSquareDeviationX = numpy.sum(pX * pX) - nCount * dMeanX * dMeanX
    dSlope = dCrossDeviationYX / dSquareDeviationX
    dIntercept = dMeanY - dSlope * dMeanX
    return dSlope, dIntercept


def sigmoid(pX: numpy.ndarray):
    return numpy.exp(-numpy.logaddexp(0, -pX))


def logistic_regression(pX: numpy.ndarray, pW: numpy.ndarray):  # Output : Calculate output (pY)
    # X is input (row-wise sample)
    # W is weight
    pArrayV = numpy.matmul(pW, pX.transpose())
    pArrayV = pArrayV.transpose()
    pArrayY = sigmoid(pArrayV)
    return pArrayY


def gradient_descent(pX: numpy.ndarray, pW: numpy.ndarray, pY: numpy.ndarray,
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
        dY = logistic_regression(iArrayX, pW)
        # backward process
        dError = dTarget - dY
        iGradient = dAlpha * dError * iArrayX
        pW = pW + iGradient.transpose()
    return pW


def back_propagation(pX: numpy.ndarray, pListW: list, pY: numpy.ndarray,
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
            iLayer = logistic_regression(iLayer.copy(), pListW[j])
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

