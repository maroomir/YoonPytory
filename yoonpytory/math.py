import numpy


def get_linear_regression(arrayX, arrayY):
    assert isinstance(arrayX, numpy.ndarray)
    assert isinstance(arrayY, numpy.ndarray)
    if arrayX.size != arrayY.size:
        raise Exception("Array X size:{0}, Y size:{1} is not equal".format(arrayX.size, arrayY.size))
    count = arrayX.size
    # Calculate Model per least-square
    mean_x, mean_y = arrayX.mean(), arrayY.mean()
    cross_deviation_yx = numpy.sum(arrayY * arrayX) - count * mean_x * mean_y
    square_deviation_x = numpy.sum(arrayX * arrayX) - count * mean_x * mean_x
    slope = cross_deviation_yx / square_deviation_x
    intercept = mean_y - slope * mean_x
    return slope, intercept
