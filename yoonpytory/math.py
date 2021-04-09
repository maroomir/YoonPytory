import numpy


def least_square(arrayX: numpy.ndarray, arrayY: numpy.ndarray):
    assert isinstance(arrayX, numpy.ndarray)
    assert isinstance(arrayY, numpy.ndarray)
    if arrayX.size != arrayY.size:
        raise Exception("Array X size:{0}, Y size:{1} is not equal".format(arrayX.size, arrayY.size))
    nCount = arrayX.size
    # Calculate Model per least-square
    meanX, meanY = arrayX.mean(), arrayY.mean()
    cross_deviation_yx = numpy.sum(arrayY * arrayX) - nCount * meanX * meanY
    square_deviation_x = numpy.sum(arrayX * arrayX) - nCount * meanX * meanX
    dSlope = cross_deviation_yx / square_deviation_x
    dIntercept = meanY - dSlope * meanX
    return dSlope, dIntercept
