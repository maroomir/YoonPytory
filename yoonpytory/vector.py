import sys
import math
import numpy
from yoonpytory.dir import eYoonDir2D


class YoonVector2D:
    x = 0
    y = 0
    __step_x = 1
    __step_y = 1

    def __init__(self, x, y, step_x=1, step_y=1):
        self.x = x
        self.y = y
        self.__step_x = step_x
        self.__step_y = step_y

    def __str__(self):
        return "{0}".format(self.to_tuple())

    @classmethod
    def direction_vector(cls, eDir: eYoonDir2D, step_x=1, step_y=1):
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
        return YoonVector2D(dic[eDir.__str__()][0] * step_x, dic[eDir.__str__()][1] * step_y, step_x, step_y)

    @classmethod
    def to_array_xy(cls, args):
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        listX, listY = [], []
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            listX.append([args[i].x])
            listY.append([args[i].y])
        return numpy.array(listX, listY)

    @classmethod
    def to_array_x(cls, args):
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        listX = []
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            listX.append([args[i].x])
        return numpy.array(listX)

    @classmethod
    def to_array_y(cls, args):
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        listY = []
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            listY.append([args[i].y])
        return numpy.array(listY)

    @classmethod
    def minimum_x(cls, args):
        if not len(args) > 0:
            raise Exception("Arguments is not enough")
        minX = sys.maxsize
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if minX > args[i].x:
                minX = args[i].x
        return minX

    @classmethod
    def maximum_x(cls, args):
        if not len(args) > 0:
            raise Exception("Arguments is not enough")
        maxX = -sys.maxsize
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if maxX < args[i].x:
                maxX = args[i].x
        return maxX

    @classmethod
    def minimum_y(cls, args):
        if not len(args) > 0:
            raise Exception("Arguments is not enough")
        minY = sys.maxsize
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if minY > args[i].y:
                minY = args[i].y
        return minY

    @classmethod
    def maximum_y(cls, args):
        if not len(args) > 0:
            raise Exception("Arguments is not enough")
        maxY = -sys.maxsize
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if maxY < args[i].x:
                maxY = args[i].x
        return maxY

    @classmethod
    def zero_vector(cls):
        return YoonVector2D(0, 0)

    def __copy__(self):
        return YoonVector2D(self.x, self.y, self.__step_x, self.__step_y)

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

    def scale(self, scaleX: (int, float), scaleY: (int, float)):
        array = numpy.eye(3)
        array[0, 0] = scaleX
        array[1, 1] = scaleY
        result = array.dot(self.to_numpy(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0], self.__step_x, self.__step_y)

    def go(self, strTag: str):
        assert isinstance(strTag, str)
        pDir = self.direction().go(strTag)
        return YoonVector2D.direction_vector(pDir, self.__step_x, self.__step_y)

    def back(self, strTag: str):
        assert isinstance(strTag, str)
        pDir = self.direction().back(strTag)
        return YoonVector2D.direction_vector(pDir, self.__step_x, self.__step_y)

    def move(self, moveX: (int, float), moveY: (int, float)):
        array = numpy.eye(3)
        array[0, 2] = moveX
        array[1, 2] = moveY
        result = array.dot(self.to_numpy(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0], self.__step_x, self.__step_y)

    def to_numpy(self, nRow: int, nCol: int):
        return numpy.array([self.x, self.y, 1]).reshape(nRow, nCol)

    def to_list(self):
        return [self.x, self.y, 1]

    def to_tuple(self):
        return self.x, self.y

    def __add__(self, other):
        assert isinstance(other, (YoonVector2D, str, eYoonDir2D))
        if isinstance(other, YoonVector2D):
            return YoonVector2D(self.x + other.x, self.y + other.y, self.__step_x, self.__step_y)
        elif isinstance(other, eYoonDir2D):
            dirVector = YoonVector2D.direction_vector(other, self.__step_x, self.__step_y)
            return YoonVector2D(self.x + dirVector.x, self.y + dirVector.y, self.__step_x, self.__step_y)
        else:
            dirVector = YoonVector2D.direction_vector(self.direction(), self.__step_x, self.__step_y)
            dirVector = dirVector.go(other)
            return YoonVector2D(self.x + dirVector.x, self.y + dirVector.y, self.__step_x, self.__step_y)

    def __sub__(self, other):
        assert isinstance(other, (YoonVector2D, str, eYoonDir2D))
        if isinstance(other, YoonVector2D):
            return YoonVector2D(self.x - other.x, self.y - other.y, self.__step_x, self.__step_y)
        elif isinstance(other, eYoonDir2D):
            dirVector = YoonVector2D.direction_vector(other, self.__step_x, self.__step_y)
            return YoonVector2D(self.x - dirVector.x, self.y - dirVector.y, self.__step_x, self.__step_y)
        else:
            dirVector = YoonVector2D.direction_vector(self.direction(), self.__step_x, self.__step_y)
            dirVector = dirVector.back(other)
            return YoonVector2D(self.x + dirVector.x, self.y + dirVector.y, self.__step_x, self.__step_y)

    def __mul__(self, number):
        assert isinstance(number, int) or isinstance(number, float)
        return YoonVector2D(self.x * number, self.y * number, self.__step_x, self.__step_y)

    def __truediv__(self, number):
        assert isinstance(number, int) or isinstance(number, float)
        return YoonVector2D(self.x / number, self.y / number, self.__step_x, self.__step_y)

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __eq__(self, other):
        assert isinstance(other, YoonVector2D)
        return self.x == other.x and self.y == other.y
