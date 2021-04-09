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


def logistic_regression(pX: numpy.ndarray, pW: numpy.ndarray): # Output : Calculate output (pY)
    # X is input (row-wise sample)
    # W is weight
    # Check Array size
    nCountX, nDimensionX = pX.shape
    nCountW, nDimensionW = pW.shape
    if nCountW != 1:
        raise Exception("The layer count is not only one")
    if nDimensionX != nDimensionW:
        raise Exception("X Dimension {0}, W Dimension {1} is not equal".format(nDimensionX, nDimensionW))
    pArrayV = numpy.matmul(pW, pX.transpose())
    pArrayY = sigmoid(pArrayV)
    return pArrayY.transpose()


def gradient_descent(pX: numpy.ndarray, pW: numpy.ndarray, pY: numpy.ndarray,
                                dAlpha=0.001):  # Output : improved weight (pW)
    # X is input
    # W is weight estimated (not constructed)
    # Y is target output (Right answer)
    nCountX, nDimensionX = pX.shape
    nCountW, nDimensionW = pW.shape
    nCountY, nDimensionY = pY.shape
    if nCountW != 1 and nDimensionY != 1:
        raise Exception("The layer count is not only one")
    if nCountX != nCountY:
        raise Exception("Array X count {0}, Delta count {1} is not equal".format(nCountX, nCountY))
    if nDimensionX != nDimensionW:
        raise Exception("X Dimension {0}, W Dimension {1} is not equal".format(nDimensionX, nDimensionW))
    for i in range(nCountX):
        iArrayX = pX[i, :].transpose()  # iInput length = nDimension
        dTarget = pY[i]  # Size = 1
        dV = numpy.matmul(pW, iArrayX)  # Size = 1
        dY = sigmoid(dV)  # Size = 1
        dError = dTarget - dY
        pErrorPerInput = dAlpha * dError * iArrayX.transpose()
        pW = pW + pErrorPerInput
    return pW
