import sys
import math
import numpy
from yoonpytory.dir import eYoonDir2D


class YoonVector2D:
    x = 0
    y = 0
    __step_x = 1
    __step_y = 1

    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        if kwargs.get("step"):
            assert isinstance(kwargs["step"], (int, float))
            self.__step_x = kwargs["step"]
            self.__step_y = kwargs["step"]
        if kwargs.get("stepX"):
            assert isinstance(kwargs["stepX"], (int, float))
            self.__step_x = kwargs["stepX"]
        if kwargs.get("stepY"):
            assert isinstance(kwargs["stepY"], (int, float))
            self.__step_y = kwargs["stepY"]

    def __str__(self):
        return "{0}".format(self.to_tuple())

    @classmethod
    def direction_vector(cls, eDir, **kwargs):
        assert isinstance(eDir, eYoonDir2D)
        dic = {eYoonDir2D.NONE: (0, 0),
               eYoonDir2D.CENTER: (0, 0),
               eYoonDir2D.TOP: (0, 1),
               eYoonDir2D.BOTTOM: (0, -1),
               eYoonDir2D.RIGHT: (1, 0),
               eYoonDir2D.LEFT: (-1, 0),
               eYoonDir2D.TOP_RIGHT: (1, 1),
               eYoonDir2D.TOP_LEFT: (-1, 1),
               eYoonDir2D.BOTTOM_LEFT: (-1, -1),
               eYoonDir2D.BOTTOM_RIGHT: (1, -1)}
        step_x = 1
        step_y = 1
        if kwargs.get("step"):
            assert isinstance(kwargs["step"], (int, float))
            step_x = kwargs["step"]
            step_y = kwargs["step"]
        if kwargs.get("stepX"):
            assert isinstance(kwargs["stepX"], (int, float))
            step_x = kwargs["stepX"]
        if kwargs.get("stepY"):
            assert isinstance(kwargs["stepY"], (int, float))
            step_y = kwargs["stepY"]
        return YoonVector2D(dic[eDir][0] * step_x, dic[eDir][1] * step_y, **kwargs)

    @classmethod
    def to_ndarray_xy(cls, args):
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        list_x, list_y = [], []
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            list_x.append([args[i].x])
            list_y.append([args[i].y])
        return numpy.array(list_x, list_y)

    @classmethod
    def to_ndarray_x(cls, args):
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        list_x = []
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            list_x.append([args[i].x])
        return numpy.array(list_x)

    @classmethod
    def to_ndarray_y(cls, args):
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        list_y = []
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            list_y.append([args[i].y])
        return numpy.array(list_y)

    @classmethod
    def minimum_x(cls, args):
        if not len(args) > 0:
            raise Exception("Arguments is not enough")
        min_x = sys.maxsize
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if min_x > args[i].x:
                min_x = args[i].x
        return min_x

    @classmethod
    def maximum_x(cls, args):
        if not len(args) > 0:
            raise Exception("Arguments is not enough")
        max_x = -sys.maxsize
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if max_x < args[i].x:
                max_x = args[i].x
        return max_x

    @classmethod
    def minimum_y(cls, args):
        if not len(args) > 0:
            raise Exception("Arguments is not enough")
        min_y = sys.maxsize
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if min_y > args[i].y:
                min_y = args[i].y
        return min_y

    @classmethod
    def maximum_y(cls, args):
        if not len(args) > 0:
            raise Exception("Arguments is not enough")
        max_y = -sys.maxsize
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if max_y < args[i].x:
                max_y = args[i].x
        return max_y

    @classmethod
    def zero_vector(cls):
        return YoonVector2D(0, 0)

    def __copy__(self):
        return YoonVector2D(self.x, self.y, stepX=self.__step_x, stepY=self.__step_y)

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

    def distance(self, vector):
        assert isinstance(vector, YoonVector2D)
        dx = self.x - vector.x
        dy = self.y - vector.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def direction_to(self, vector):
        assert isinstance(vector, YoonVector2D)
        diff_vector = self - vector
        return diff_vector.direction()

    def angle(self, vector):
        assert isinstance(vector, YoonVector2D)
        dx = vector.x - self.x
        dy = vector.y - self.y
        return math.atan2(dy, dx)

    def scale(self, scaleX, scaleY):
        matrix = numpy.eye(3)
        matrix[0, 0] = scaleX
        matrix[1, 1] = scaleY
        result = matrix.dot(self.to_numpy(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0], stepX=self.__step_x, stepY=self.__step_y)

    def go(self, strTag):
        assert isinstance(strTag, str)
        direction = self.direction().go(strTag)
        return YoonVector2D.direction_vector(direction, stepX=self.__step_x, stepY=self.__step_y)

    def back(self, strTag):
        assert isinstance(strTag, str)
        direction = self.direction().back(strTag)
        return YoonVector2D.direction_vector(direction, stepX=self.__step_x, stepY=self.__step_y)

    def move(self, moveX, moveY):
        matrix = numpy.eye(3)
        matrix[0, 2] = moveX
        matrix[1, 2] = moveY
        result = matrix.dot(self.to_numpy(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0], stepX=self.__step_x, stepY=self.__step_y)

    def to_numpy(self, nRow, nCol):
        return numpy.array([self.x, self.y, 1]).reshape(nRow, nCol)

    def to_list(self):
        return [self.x, self.y, 1]

    def to_tuple(self):
        return self.x, self.y

    def __add__(self, other):
        assert isinstance(other, (YoonVector2D, str, eYoonDir2D))
        if isinstance(other, YoonVector2D):
            return YoonVector2D(self.x + other.x, self.y + other.y, stepX=self.__step_x, stepY=self.__step_y)
        elif isinstance(other, eYoonDir2D):
            dir_vector = YoonVector2D.direction_vector(other, stepX=self.__step_x, stepY=self.__step_y)
            return YoonVector2D(self.x + dir_vector.x, self.y + dir_vector.y, stepX=self.__step_x, stepY=self.__step_y)
        else:
            dir_vector = YoonVector2D.direction_vector(self.direction(), stepX=self.__step_x, stepY=self.__step_y)
            dir_vector = dir_vector.go(other)
            return YoonVector2D(self.x + dir_vector.x, self.y + dir_vector.y, stepX=self.__step_x, stepY=self.__step_y)

    def __sub__(self, other):
        assert isinstance(other, (YoonVector2D, str, eYoonDir2D))
        if isinstance(other, YoonVector2D):
            return YoonVector2D(self.x - other.x, self.y - other.y, stepX=self.__step_x, stepY=self.__step_y)
        elif isinstance(other, eYoonDir2D):
            dir_vector = YoonVector2D.direction_vector(other, stepX=self.__step_x, stepY=self.__step_y)
            return YoonVector2D(self.x - dir_vector.x, self.y - dir_vector.y, stepX=self.__step_x, stepY=self.__step_y)
        else:
            dir_vector = YoonVector2D.direction_vector(self.direction(), stepX=self.__step_x, stepY=self.__step_y)
            dir_vector = dir_vector.back(other)
            return YoonVector2D(self.x + dir_vector.x, self.y + dir_vector.y, stepX=self.__step_x, stepY=self.__step_y)

    def __mul__(self, number):
        assert isinstance(number, int) or isinstance(number, float)
        return YoonVector2D(self.x * number, self.y * number, stepX=self.__step_x, stepY=self.__step_y)

    def __truediv__(self, number):
        assert isinstance(number, int) or isinstance(number, float)
        return YoonVector2D(self.x / number, self.y / number, stepX=self.__step_x, stepY=self.__step_y)

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __eq__(self, other):
        assert isinstance(other, YoonVector2D)
        return self.x == other.x and self.y == other.y