import math
import sys

import numpy

from yoonpytory.dir import eYoonDir2D


def _list_square(pX: numpy.ndarray, pY: numpy.ndarray):
    if pX.size != pY.size:
        raise Exception("Array X size:{0}, Y size:{1} is not equal".format(pX.size, pY.size))
    nCount = pX.size
    # Calculate Model per least-square
    dMeanX, dMeanY = pX.mean(), pY.mean()
    dCrossDeviationYX = numpy.sum(pY * pX) - nCount * dMeanX * dMeanY
    dSquareDeviationX = numpy.sum(pX * pX) - nCount * dMeanX * dMeanX
    dSquareDeviationX = 1e-10 if dSquareDeviationX < 1e-10 else dSquareDeviationX
    dSlope = dCrossDeviationYX / dSquareDeviationX
    dIntercept = dMeanY - dSlope * dMeanX
    return dSlope, dIntercept


class YoonVector2D:
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    """
    def __init__(self, dX=0.0, dY=0.0, nStepX=1, nStepY=1):
        self.x = dX
        self.y = dY
        self.__stepX = nStepX
        self.__stepY = nStepY

    def __str__(self):
        return "{0}".format(self.to_tuple())

    @classmethod
    def direction_vector(cls, eDir: eYoonDir2D, nStepX=1, nStepY=1):
        assert isinstance(eDir, eYoonDir2D)
        dic = {"NONE": (0, 0),
               "CENTER": (0, 0),
               "TOP": (0, 1),
               "BOTTOM": (0, -1),
               "RIGHT": (1, 0),
               "LEFT": (-1, 0),
               "TOP_RIGHT": (1, 1),
               "TOP_LEFT": (-1, 1),
               "BOTTOM_LEFT": (-1, -1),
               "BOTTOM_RIGHT": (1, -1)}
        return YoonVector2D(dic[eDir.__str__()][0] * nStepX, dic[eDir.__str__()][1] * nStepY, nStepX, nStepY)

    @classmethod
    def list_to_array_xy(cls, pList: list, *args):
        pListX, pListY = [], []
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
                pListX.append(args[i].x)
                pListY.append(args[i].y)
            return numpy.array(pListX, pListY)
        else:
            if not len(pList) >= 2:
                raise Exception("Arguments is not enough")
            for i in range(len(pList)):
                assert isinstance(pList[i], YoonVector2D)
                pListX.append([pList[i].x])
                pListY.append([pList[i].y])
            return numpy.array(pListX, pListY)

    @classmethod
    def list_to_array_x(cls, pList: list, *args):
        pListX = []
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
                pListX.append([args[i].x])
            return numpy.array(pListX)
        else:
            if not len(pList) >= 2:
                raise Exception("Arguments is not enough")
            for i in range(len(pList)):
                assert isinstance(pList[i], YoonVector2D)
                pListX.append([pList[i].x])
            return numpy.array(pListX)

    @classmethod
    def list_to_array_y(cls, pList: list, *args):
        listY = []
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
                listY.append([args[i].y])
            return numpy.array(listY)
        else:
            if not len(pList) >= 2:
                raise Exception("Arguments is not enough")
            for i in range(len(pList)):
                assert isinstance(pList[i], YoonVector2D)
                listY.append([pList[i].y])
            return numpy.array(listY)

    @classmethod
    def list_to_minimum_x(cls, pList: list, *args):
        minX = sys.maxsize
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
                if minX > args[i].x:
                    minX = args[i].x
            return minX
        else:
            if not len(pList) > 0:
                raise Exception("Arguments is not enough")
            for i in range(len(pList)):
                assert isinstance(pList[i], YoonVector2D)
                if minX > pList[i].x:
                    minX = pList[i].x
            return minX

    @classmethod
    def list_to_maximum_x(cls, pList: list, *args):
        maxX = -sys.maxsize
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
                if maxX < args[i].x:
                    maxX = args[i].x
            return maxX
        else:
            if not len(pList) > 0:
                raise Exception("Arguments is not enough")
            for i in range(len(pList)):
                assert isinstance(pList[i], YoonVector2D)
                if maxX < pList[i].x:
                    maxX = pList[i].x
            return maxX

    @classmethod
    def list_to_minimum_y(cls, pList: list, *args):
        minY = sys.maxsize
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
                if minY > args[i].y:
                    minY = args[i].y
            return minY
        else:
            if not len(pList) > 0:
                raise Exception("Arguments is not enough")
            for i in range(len(pList)):
                assert isinstance(pList[i], YoonVector2D)
                if minY > pList[i].y:
                    minY = pList[i].y
            return minY

    @classmethod
    def list_to_maximum_y(cls, pList: list, *args):
        maxY = -sys.maxsize
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
                if maxY < args[i].x:
                    maxY = args[i].x
            return maxY
        else:
            if not len(pList) > 0:
                raise Exception("Arguments is not enough")
            for i in range(len(pList)):
                assert isinstance(pList[i], YoonVector2D)
                if maxY < pList[i].x:
                    maxY = pList[i].x
            return maxY

    @classmethod
    def zero_vector(cls):
        return YoonVector2D(0, 0)

    @classmethod
    def from_array(cls, pArray):
        assert isinstance(pArray, (tuple, list, numpy.array))
        return YoonVector2D(pArray[0], pArray[1])

    def __copy__(self):
        return YoonVector2D(self.x, self.y, self.__stepX, self.__stepY)

    def zero(self):
        self.x = 0
        self.y = 0

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def unit(self):
        if self.length() > 0.0001:
            length = 1.0 / self.length()
            self.x *= length
            self.y *= length

    def direction(self):
        if self.x == 0 and self.y == 0:
            return eYoonDir2D.CENTER
        elif self.x == 0 and self.y > 0:
            return eYoonDir2D.TOP
        elif self.x == 0 and self.y < 0:
            return eYoonDir2D.BOTTOM
        elif self.x > 0 and self.y == 0:
            return eYoonDir2D.RIGHT
        elif self.x < 0 and self.y == 0:
            return eYoonDir2D.LEFT
        elif self.x > 0 and self.y > 0:
            return eYoonDir2D.TOP_RIGHT
        elif self.x < 0 < self.y:
            return eYoonDir2D.TOP_LEFT
        elif self.x < 0 and self.y < 0:
            return eYoonDir2D.BOTTOM_LEFT
        elif self.x > 0 > self.y:
            return eYoonDir2D.BOTTOM_RIGHT
        else:
            return eYoonDir2D.NONE

    def distance(self, pVector):
        assert isinstance(pVector, YoonVector2D)
        dx = self.x - pVector.x
        dy = self.y - pVector.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def direction_to(self, pVector):
        assert isinstance(pVector, YoonVector2D)
        diff_vector = self - pVector
        return diff_vector.direction()

    def angle(self, pVector):
        assert isinstance(pVector, YoonVector2D)
        dx = pVector.x - self.x
        dy = pVector.y - self.y
        return math.atan2(dy, dx)

    def scale(self, dScaleX: (int, float), dScaleY: (int, float)):
        array = numpy.eye(3)
        array[0, 0] = dScaleX
        array[1, 1] = dScaleY
        result = array.dot(self.to_array(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0], self.__stepX, self.__stepY)

    def go(self, strTag: str):
        assert isinstance(strTag, str)
        pDir = self.direction().go(strTag)
        return YoonVector2D.direction_vector(pDir, self.__stepX, self.__stepY)

    def back(self, strTag: str):
        assert isinstance(strTag, str)
        pDir = self.direction().back(strTag)
        return YoonVector2D.direction_vector(pDir, self.__stepX, self.__stepY)

    def move(self, dMoveX: (int, float), dMoveY: (int, float)):
        array = numpy.eye(3)
        array[0, 2] = dMoveX
        array[1, 2] = dMoveY
        result = array.dot(self.to_array(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0], self.__stepX, self.__stepY)

    def to_array(self, nRow: int, nCol: int):
        return numpy.array([self.x, self.y, 1]).reshape(nRow, nCol)

    def to_list(self):
        return [self.x, self.y, 1]

    def to_list_int(self):
        return [int(self.x), int(self.y), 1]

    def to_tuple(self):
        return self.x, self.y

    def to_tuple_int(self):
        dX = 0 if math.isnan(self.x) else self.x
        dY = 0 if math.isnan(self.y) else self.y
        return int(dX), int(dY)

    def __add__(self, other):
        assert isinstance(other, (YoonVector2D, str, eYoonDir2D))
        if isinstance(other, YoonVector2D):
            return YoonVector2D(self.x + other.x, self.y + other.y, self.__stepX, self.__stepY)
        elif isinstance(other, eYoonDir2D):
            dirVector = YoonVector2D.direction_vector(other, self.__stepX, self.__stepY)
            return YoonVector2D(self.x + dirVector.x, self.y + dirVector.y, self.__stepX, self.__stepY)
        else:
            dirVector = YoonVector2D.direction_vector(self.direction(), self.__stepX, self.__stepY)
            dirVector = dirVector.go(other)
            return YoonVector2D(self.x + dirVector.x, self.y + dirVector.y, self.__stepX, self.__stepY)

    def __sub__(self, other):
        assert isinstance(other, (YoonVector2D, str, eYoonDir2D))
        if isinstance(other, YoonVector2D):
            return YoonVector2D(self.x - other.x, self.y - other.y, self.__stepX, self.__stepY)
        elif isinstance(other, eYoonDir2D):
            dirVector = YoonVector2D.direction_vector(other, self.__stepX, self.__stepY)
            return YoonVector2D(self.x - dirVector.x, self.y - dirVector.y, self.__stepX, self.__stepY)
        else:
            dirVector = YoonVector2D.direction_vector(self.direction(), self.__stepX, self.__stepY)
            dirVector = dirVector.back(other)
            return YoonVector2D(self.x + dirVector.x, self.y + dirVector.y, self.__stepX, self.__stepY)

    def __mul__(self, number):
        assert isinstance(number, int) or isinstance(number, float)
        return YoonVector2D(self.x * number, self.y * number, self.__stepX, self.__stepY)

    def __truediv__(self, number):
        assert isinstance(number, int) or isinstance(number, float)
        return YoonVector2D(self.x / number, self.y / number, self.__stepX, self.__stepY)

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __eq__(self, other):
        assert isinstance(other, YoonVector2D)
        return self.x == other.x and self.y == other.y


class YoonLine2D:
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    """

    @classmethod
    def from_list(cls,
                  pList: list):
        pLine = YoonLine2D()
        if pList is not None:
            pArrayX = YoonVector2D.list_to_array_x(pList)
            pArrayY = YoonVector2D.list_to_array_y(pList)
            dMinX = numpy.min(pArrayX)
            dMaxX = numpy.max(pArrayX)
            pLine.slope, pLine.intercept = _list_square(pArrayX, pArrayY)
            pLine.startPos = YoonVector2D(dMinX, pLine.y(dMinX))
            pLine.endPos = YoonVector2D(dMaxX, pLine.y(dMaxX))
        return pLine

    @classmethod
    def from_vectors(cls,
                     *args: YoonVector2D):
        pLine = YoonLine2D()
        if len(args) > 0:
            pArrayX = YoonVector2D.list_to_array_x(args)
            pArrayY = YoonVector2D.list_to_array_y(args)
            dMinX = YoonVector2D.list_to_minimum_x(args)
            dMinY = YoonVector2D.list_to_minimum_y(args)
            pLine.slope, pLine.intercept = _list_square(pArrayX, pArrayY)
            pLine.startPos = YoonVector2D(dMinX, pLine.y(dMinX))
            pLine.endPos = YoonVector2D(dMinY, pLine.y(dMinY))
        return pLine

    def __init__(self,
                 dSlope=0.0,
                 dIntercept=0.0,
                 pStartVector: YoonVector2D = None,
                 pEndVector: YoonVector2D = None):
        self.slope = dSlope
        self.intercept = dIntercept
        self.startPos = pStartVector
        self.endPos = pEndVector

    def __str__(self):
        return "SLOPE : {0}, INTERCEPT : {1}".format(self.slope, self.intercept)

    def __copy__(self):
        return YoonLine2D(dSlope=self.slope,
                          dIntercept=self.intercept,
                          pStartVector=self.startPos,
                          pEndVector=self.endPos)

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


class YoonRect2D:
    # The shared area of YoonDataset class
    # All of instances are using this shared area
    def __init__(self,
                 pList: list = None,
                 pPos: YoonVector2D = None,
                 dX=0.0,
                 dY=0.0,
                 dWidth=0.0,
                 dHeight=0.0,
                 *args: YoonVector2D,
                 **kwargs):
        if len(args) > 0:
            dMinX = dMinY = sys.maxsize
            dMaxX = dMaxY = -sys.maxsize
            for pVector in args:
                assert isinstance(pVector, YoonVector2D)
                if pVector.x < dMinX:
                    dMinX = pVector.x
                elif pVector.x > dMaxX:
                    dMaxX = pVector.x
                if pVector.y < dMinY:
                    dMinY = pVector.y
                elif pVector.y > dMaxY:
                    dMaxY = pVector.y
            self.centerPos.x = (dMinX + dMaxX) / 2
            self.centerPos.y = (dMinY + dMaxY) / 2
            self.width = dMaxX - dMinX
            self.height = dMaxY - dMinY
        elif kwargs.get("dir1") and kwargs.get("dir2") and kwargs.get("pos1") and kwargs.get("pos2"):
            assert isinstance(kwargs["dir1"], eYoonDir2D)
            assert isinstance(kwargs["dir2"], eYoonDir2D)
            dir1 = kwargs["dir1"]
            dir2 = kwargs["dir2"]
            pos1 = kwargs["pos1"]
            pos2 = kwargs["pos2"]
            if dir1 == eYoonDir2D.TOP_LEFT and dir2 == eYoonDir2D.BOTTOM_RIGHT:
                self.centerPos = (pos1 + pos2) / 2
                self.width = (pos2 - pos1).x
                self.height = (pos2 - pos1).y
            elif dir1 == eYoonDir2D.BOTTOM_RIGHT and dir2 == eYoonDir2D.TOP_LEFT:
                self.centerPos = (pos1 + pos2) / 2
                self.width = (pos1 - pos2).x
                self.height = (pos1 - pos2).y
            elif dir1 == eYoonDir2D.TOP_RIGHT and dir2 == eYoonDir2D.BOTTOM_RIGHT:
                self.centerPos = (pos1 + pos2) / 2
                self.width = (pos2 - pos1).x
                self.height = (pos1 - pos2).y
            elif dir1 == eYoonDir2D.BOTTOM_LEFT and dir2 == eYoonDir2D.TOP_RIGHT:
                self.centerPos = (pos1 + pos2) / 2
                self.width = (pos1 - pos2).x
                self.height = (pos2 - pos1).y
        else:
            if pList is not None:
                dMinX = dMinY = sys.maxsize
                dMaxX = dMaxY = -sys.maxsize
                for pVector in pList:
                    assert isinstance(pVector, YoonVector2D)
                    if pVector.x < dMinX:
                        dMinX = pVector.x
                    elif pVector.x > dMaxX:
                        dMaxX = pVector.x
                    if pVector.y < dMinY:
                        dMinY = pVector.y
                    elif pVector.y > dMaxY:
                        dMaxY = pVector.y
                self.centerPos.x = (dMinX + dMaxX) / 2
                self.centerPos.y = (dMinY + dMaxY) / 2
                self.width = dMaxX - dMinX
                self.height = dMaxY - dMinY
            else:
                if pPos is not None:
                    self.centerPos = pPos.__copy__()
                else:
                    self.centerPos = YoonVector2D(dX, dY).__copy__()
                self.width = dWidth
                self.height = dHeight

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, CENTER {2}, TL : {3}, BR : {4}".format(self.width, self.height,
                                                                                  self.centerPos.__str__(),
                                                                                  self.top_left().__str__(),
                                                                                  self.bottom_right().__str__())

    def __copy__(self):
        return YoonRect2D(pPos=self.centerPos, dWidth=self.width, dHeight=self.height)

    def left(self):
        return self.centerPos.x - self.width / 2

    def top(self):
        return self.centerPos.y - self.height / 2

    def right(self):
        return self.centerPos.x + self.width / 2

    def bottom(self):
        return self.centerPos.y + self.height / 2

    def top_left(self):
        return YoonVector2D(self.left(), self.top())

    def top_right(self):
        return YoonVector2D(self.right(), self.top())

    def bottom_left(self):
        return YoonVector2D(self.left(), self.bottom())

    def bottom_right(self):
        return YoonVector2D(self.right(), self.bottom())

    def to_array_x(self):
        return numpy.array([self.top_right().x, self.top_left().x, self.bottom_left().x, self.bottom_right().x])

    def to_array_y(self):
        return numpy.array([self.top_right().y, self.top_left().y, self.bottom_left().y, self.bottom_right().y])

    def to_array_xy(self):
        pListX = [self.top_right().x, self.top_left().x, self.bottom_left().x, self.bottom_right().x]
        pListY = [self.top_right().y, self.top_left().y, self.bottom_left().y, self.bottom_right().y]
        return numpy.array(pListX, pListY)

    def to_list(self):
        return [self.left(), self.top(), self.width, self.height]

    def to_tuple(self):
        return self.left(), self.top(), self.width, self.height

    def area(self):
        return self.width * self.height

    def is_contain(self, pPos: YoonVector2D):
        assert isinstance(pPos, YoonVector2D)
        if self.left() < pPos.x < self.right() and self.top() < pPos.y < self.bottom():
            return True
        else:
            return False

    def __add__(self, pRect):
        assert isinstance(pRect, YoonRect2D)
        top = min(self.top(), pRect.top(), self.bottom(), pRect.bottom())
        bottom = max(self.top(), pRect.top(), self.bottom(), pRect.bottom())
        left = min(self.left(), pRect.left(), self.right(), pRect.right())
        right = max(self.left(), pRect.left(), self.right(), pRect.right())
        return YoonRect2D(dX=(left + right) / 2, dY=(top + bottom) / 2, dWidth=right - left, dHeight=bottom - top)

    def __mul__(self, dScale: (int, float)):
        assert isinstance(dScale, (int, float))
        return YoonRect2D(pPos=self.centerPos.__copy__(), dWidth=dScale * self.width, dHeight=dScale * self.height)

    def __eq__(self, pRect: (int, float)):
        assert isinstance(pRect, YoonRect2D)
        return self.centerPos == pRect.centerPos and self.width == pRect.width and self.height == pRect.height
