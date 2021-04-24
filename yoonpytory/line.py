import math
import numpy
from yoonpytory.vector import YoonVector2D


def _list_square(pX: numpy.ndarray, pY: numpy.ndarray):
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


class YoonLine2D:
    slope: (int, float)
    intercept: (int, float)
    startPos = YoonVector2D(0, 0).__copy__()
    endPos = YoonVector2D(0, 0).__copy__()

    def __str__(self):
        return "SLOPE : {0}, INTERCEPT : {1}".format(self.slope, self.intercept)

    def __init__(self,
                 pList: list = None,
                 dSlope=0.0,
                 dIntercept=0.0,
                 *args: YoonVector2D,
                 **kwargs):
        if len(args) > 0:
            pArrayX = YoonVector2D.list_to_array_x(args)
            pArrayY = YoonVector2D.list_to_array_y(args)
            dMinX = YoonVector2D.list_to_minimum_x(args)
            dMinY = YoonVector2D.list_to_minimum_y(args)
            self.slope, self.intercept = _list_square(pArrayX, pArrayY)
            self.startPos = YoonVector2D(dMinX, self.y(dMinX)).__copy__()
            self.endPos = YoonVector2D(dMinY, self.y(dMinY)).__copy__()
        elif kwargs.get("x1") and kwargs.get("x2") and kwargs.get("y1") and kwargs.get("y2"):
            assert isinstance(kwargs["x1"], (int, float))
            assert isinstance(kwargs["x2"], (int, float))
            assert isinstance(kwargs["y1"], (int, float))
            assert isinstance(kwargs["y2"], (int, float))
            self.slope = (kwargs["y1"] - kwargs["y2"]) / (kwargs["x1"] - kwargs["x2"])
            self.intercept = kwargs["y1"] - self.slope * kwargs["x1"]
            dMinX = kwargs["x1"] if kwargs["x1"] < kwargs["x2"] else kwargs["x2"]
            dMinY = kwargs["x1"] if kwargs["x1"] > kwargs["x2"] else kwargs["x2"]
            self.startPos = YoonVector2D(dMinX, self.y(dMinX)).__copy__()
            self.endPos = YoonVector2D(dMinY, self.y(dMinY)).__copy__()
        else:
            if pList is not None:
                pArrayX = YoonVector2D.list_to_array_x(pList)
                pArrayY = YoonVector2D.list_to_array_y(pList)
                self.slope, self.intercept = _list_square(pArrayX, pArrayY)
            else:
                self.slope = dSlope
                self.intercept = dIntercept

    def __copy__(self):
        return YoonLine2D(dSlope=self.slope, dIntercept=self.intercept)

    def x(self, dY: (int, float)):
        assert isinstance(dY, (int, float))
        return (dY - self.intercept) / self.slope

    def y(self, dX: (int, float)):
        assert isinstance(dX, (int, float))
        return dX * self.slope + self.intercept

    def distance(self, pPos: YoonVector2D):
        assert isinstance(pPos, YoonVector2D)
        return abs(self.slope * pPos.x - pPos.y + self.intercept) / math.sqrt(self.slope ** 2 + 1)

    def is_contain(self, pPos: YoonVector2D):
        assert isinstance(pPos, YoonVector2D)
        return pPos.y == pPos.x * self.slope + self.intercept

    def __add__(self, pLine):
        assert isinstance(pLine, YoonLine2D)
        return YoonLine2D(dSlope=self.slope + pLine.slope, dIntercept=self.intercept + pLine.intercept)

    def __sub__(self, pLine):
        assert isinstance(pLine, YoonLine2D)
        return YoonLine2D(dSlope=self.slope - pLine.slope, dIntercept=self.intercept + pLine.intercept)

    def __eq__(self, pLine):
        assert isinstance(pLine, YoonLine2D)
        return (self.slope == pLine.slope) and (self.intercept == pLine.intercept)
