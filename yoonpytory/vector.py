import sys
import math
import numpy
from yoonpytory.dir import eYoonDir2D


class YoonVector2D:
    x = 0
    y = 0
    __stepX = 1
    __stepY = 1

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
    def to_array_xy(cls, pArgs):
        if not len(pArgs) >= 2:
            raise Exception("Arguments is not enough")
        listX, listY = [], []
        for i in range(len(pArgs)):
            assert isinstance(pArgs[i], YoonVector2D)
            listX.append([pArgs[i].x])
            listY.append([pArgs[i].y])
        return numpy.array(listX, listY)

    @classmethod
    def to_array_x(cls, pArgs):
        if not len(pArgs) >= 2:
            raise Exception("Arguments is not enough")
        listX = []
        for i in range(len(pArgs)):
            assert isinstance(pArgs[i], YoonVector2D)
            listX.append([pArgs[i].x])
        return numpy.array(listX)

    @classmethod
    def to_array_y(cls, pArgs):
        if not len(pArgs) >= 2:
            raise Exception("Arguments is not enough")
        listY = []
        for i in range(len(pArgs)):
            assert isinstance(pArgs[i], YoonVector2D)
            listY.append([pArgs[i].y])
        return numpy.array(listY)

    @classmethod
    def minimum_x(cls, pArgs):
        if not len(pArgs) > 0:
            raise Exception("Arguments is not enough")
        minX = sys.maxsize
        for i in range(len(pArgs)):
            assert isinstance(pArgs[i], YoonVector2D)
            if minX > pArgs[i].x:
                minX = pArgs[i].x
        return minX

    @classmethod
    def maximum_x(cls, pArgs):
        if not len(pArgs) > 0:
            raise Exception("Arguments is not enough")
        maxX = -sys.maxsize
        for i in range(len(pArgs)):
            assert isinstance(pArgs[i], YoonVector2D)
            if maxX < pArgs[i].x:
                maxX = pArgs[i].x
        return maxX

    @classmethod
    def minimum_y(cls, pArgs):
        if not len(pArgs) > 0:
            raise Exception("Arguments is not enough")
        minY = sys.maxsize
        for i in range(len(pArgs)):
            assert isinstance(pArgs[i], YoonVector2D)
            if minY > pArgs[i].y:
                minY = pArgs[i].y
        return minY

    @classmethod
    def maximum_y(cls, pArgs):
        if not len(pArgs) > 0:
            raise Exception("Arguments is not enough")
        maxY = -sys.maxsize
        for i in range(len(pArgs)):
            assert isinstance(pArgs[i], YoonVector2D)
            if maxY < pArgs[i].x:
                maxY = pArgs[i].x
        return maxY

    @classmethod
    def zero_vector(cls):
        return YoonVector2D(0, 0)

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
        result = array.dot(self.to_numpy(3, 1))
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
        result = array.dot(self.to_numpy(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0], self.__stepX, self.__stepY)

    def to_numpy(self, nRow: int, nCol: int):
        return numpy.array([self.x, self.y, 1]).reshape(nRow, nCol)

    def to_list(self):
        return [self.x, self.y, 1]

    def to_tuple(self):
        return self.x, self.y

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
