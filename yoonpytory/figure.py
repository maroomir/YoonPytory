import math
import sys

import numpy
from numpy import ndarray

from yoonpytory.dir import YoonDir2D


class YoonVector2D:
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    """

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __str__(self):
        return "{0}".format(self.to_tuple())

    @classmethod
    def direction_vector(cls,
                         dir_: YoonDir2D,
                         step_x=1, step_y=1):
        assert isinstance(dir_, YoonDir2D)
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
        return YoonVector2D(dic[dir_.__str__()][0] * step_x, dic[dir_.__str__()][1] * step_y)

    @classmethod
    def list_to_array_xy(cls, list_: list):
        list_x, list_y = [], []
        if not len(list_) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(list_)):
            assert isinstance(list_[i], YoonVector2D)
            list_x.append([list_[i].x])
            list_y.append([list_[i].y])
        return numpy.array([list_x, list_y]).transpose()

    @classmethod
    def vectors_to_array_xy(cls, *args):
        list_x, list_y = [], []
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            list_x.append(args[i].x)
            list_y.append(args[i].y)
        return numpy.array([list_x, list_y]).transpose()

    @classmethod
    def list_to_array_x(cls, list_: list):
        list_x = []
        if not len(list_) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(list_)):
            assert isinstance(list_[i], YoonVector2D)
            list_x.append([list_[i].x])
        return numpy.array(list_x)

    @classmethod
    def vectors_to_array_x(cls, *args):
        list_x = []
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            list_x.append([args[i].x])
        return numpy.array(list_x)

    @classmethod
    def list_to_array_y(cls, list_: list):
        list_y = []
        if not len(list_) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(list_)):
            assert isinstance(list_[i], YoonVector2D)
            list_y.append([list_[i].y])
        return numpy.array(list_y)

    @classmethod
    def vectors_to_array_y(cls, *args):
        list_y = []
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            list_y.append([args[i].y])
        return numpy.array(list_y)

    @classmethod
    def minimum_x_in_list(cls, list_: list):
        min_x = sys.maxsize
        if not len(list_) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(list_)):
            assert isinstance(list_[i], YoonVector2D)
            if min_x > list_[i].x:
                min_x = list_[i].x
        return min_x

    @classmethod
    def minimum_x_in_vectors(cls, *args):
        min_x = sys.maxsize
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if min_x > args[i].x:
                min_x = args[i].x
        return min_x

    @classmethod
    def maximum_x_in_list(cls, list_: list):
        max_x = -sys.maxsize
        if not len(list_) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(list_)):
            assert isinstance(list_[i], YoonVector2D)
            if max_x < list_[i].x:
                max_x = list_[i].x
        return max_x

    @classmethod
    def maximum_x_in_vectors(cls, *args):
        max_x = -sys.maxsize
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if max_x < args[i].x:
                max_x = args[i].x
        return max_x

    @classmethod
    def minimum_y_in_list(cls, list_: list):
        min_y = sys.maxsize
        if not len(list_) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(list_)):
            assert isinstance(list_[i], YoonVector2D)
            if min_y > list_[i].y:
                min_y = list_[i].y
        return min_y

    @classmethod
    def minimum_y_in_vectors(cls, *args):
        min_y = sys.maxsize
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if min_y > args[i].y:
                min_y = args[i].y
        return min_y

    @classmethod
    def maximum_y_in_list(cls, list_: list):
        max_y = -sys.maxsize
        if not len(list_) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(list_)):
            assert isinstance(list_[i], YoonVector2D)
            if max_y < list_[i].x:
                max_y = list_[i].x
        return max_y

    @classmethod
    def maximum_y_in_vectors(cls, *args):
        max_y = -sys.maxsize
        if not len(args) >= 2:
            raise Exception("Arguments is not enough")
        for i in range(len(args)):
            assert isinstance(args[i], YoonVector2D)
            if max_y < args[i].x:
                max_y = args[i].x
        return max_y

    @classmethod
    def zero_vector(cls):
        return YoonVector2D(0, 0)

    @classmethod
    def from_array(cls, array):
        assert isinstance(array, (tuple, list, numpy.array))
        return YoonVector2D(array[0], array[1])

    def __copy__(self):
        return YoonVector2D(self.x, self.y)

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
            return YoonDir2D.CENTER
        elif self.x == 0 and self.y > 0:
            return YoonDir2D.TOP
        elif self.x == 0 and self.y < 0:
            return YoonDir2D.BOTTOM
        elif self.x > 0 and self.y == 0:
            return YoonDir2D.RIGHT
        elif self.x < 0 and self.y == 0:
            return YoonDir2D.LEFT
        elif self.x > 0 and self.y > 0:
            return YoonDir2D.TOP_RIGHT
        elif self.x < 0 < self.y:
            return YoonDir2D.TOP_LEFT
        elif self.x < 0 and self.y < 0:
            return YoonDir2D.BOTTOM_LEFT
        elif self.x > 0 > self.y:
            return YoonDir2D.BOTTOM_RIGHT
        else:
            return YoonDir2D.NONE

    def distance(self, vector):
        assert isinstance(vector, YoonVector2D)
        dist_x = self.x - vector.x
        dist_y = self.y - vector.y
        return math.sqrt(dist_x ** 2 + dist_y ** 2)

    def direction_to(self, vector):
        assert isinstance(vector, YoonVector2D)
        diff_vector = self - vector
        return diff_vector.direction()

    def angle(self, vector):
        assert isinstance(vector, YoonVector2D)
        dist_x = vector.x - self.x
        dist_y = vector.y - self.y
        return math.atan2(dist_y, dist_x)

    def scale(self, scale_x: (int, float), scale_y: (int, float)):
        array = numpy.eye(3)
        array[0, 0] = scale_x
        array[1, 1] = scale_y
        result = array.dot(self.to_array(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0])

    def go(self, tag: str, step_x=1, step_y=1):
        assert isinstance(tag, str)
        _dir = self.direction().go(tag)
        return YoonVector2D.direction_vector(_dir, step_x, step_y)

    def back(self, tag: str, step_x=1, step_y=1):
        assert isinstance(tag, str)
        _dir = self.direction().back(tag)
        return YoonVector2D.direction_vector(_dir, step_x, step_y)

    def move(self, move_x: (int, float), move_y: (int, float)):
        array = numpy.eye(3)
        array[0, 2] = move_x
        array[1, 2] = move_y
        result = array.dot(self.to_array(3, 1))
        return YoonVector2D(result[0, 0], result[1, 0])

    def to_array(self, row: int, col: int):
        return numpy.array([self.x, self.y, 1]).reshape(row, col)

    def to_list(self):
        return [self.x, self.y, 1]

    def to_list_int(self):
        return [int(self.x), int(self.y), 1]

    def to_tuple(self):
        return self.x, self.y

    def to_tuple_int(self):
        x = 0 if math.isnan(self.x) else self.x
        y = 0 if math.isnan(self.y) else self.y
        return int(x), int(y)

    def __add__(self, other):
        assert isinstance(other, (YoonVector2D, str, YoonDir2D))
        if isinstance(other, YoonVector2D):
            return YoonVector2D(self.x + other.x, self.y + other.y)
        elif isinstance(other, YoonDir2D):
            dir_vector = YoonVector2D.direction_vector(other)
            return YoonVector2D(self.x + dir_vector.x, self.y + dir_vector.y)
        else:
            dir_vector = YoonVector2D.direction_vector(self.direction())
            dir_vector = dir_vector.go(other)
            return YoonVector2D(self.x + dir_vector.x, self.y + dir_vector.y)

    def __sub__(self, other):
        assert isinstance(other, (YoonVector2D, str, YoonDir2D))
        if isinstance(other, YoonVector2D):
            return YoonVector2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, YoonDir2D):
            dir_vector = YoonVector2D.direction_vector(other)
            return YoonVector2D(self.x - dir_vector.x, self.y - dir_vector.y)
        else:
            dir_vector = YoonVector2D.direction_vector(self.direction())
            dir_vector = dir_vector.back(other)
            return YoonVector2D(self.x + dir_vector.x, self.y + dir_vector.y)

    def __mul__(self, number):
        assert isinstance(number, int) or isinstance(number, float)
        return YoonVector2D(self.x * number, self.y * number)

    def __truediv__(self, number):
        assert isinstance(number, int) or isinstance(number, float)
        return YoonVector2D(self.x / number, self.y / number)

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
    def _list_square(cls,
                     array_x: numpy.ndarray, array_y: numpy.ndarray):
        if array_x.size != array_y.size:
            raise Exception("Array X size:{0}, Y size:{1} is not equal".format(array_x.size, array_y.size))
        count = array_x.size
        # Calculate Model per least-square
        mean_x, mean_y = array_x.mean(), array_y.mean()
        cross_deviation_yx = numpy.sum(array_y * array_x) - count * mean_x * mean_y
        square_deviation_x = numpy.sum(array_x * array_y) - count * mean_x * mean_x
        square_deviation_x = 1e-10 if square_deviation_x < 1e-10 else square_deviation_x
        slope = cross_deviation_yx / square_deviation_x
        intercept = mean_y - slope * mean_x
        return slope, intercept

    @classmethod
    def from_list(cls,
                  args: list):
        line = YoonLine2D()
        if args is not None:
            array_x = YoonVector2D.list_to_array_x(args)
            array_y = YoonVector2D.list_to_array_y(args)
            min_x = numpy.min(array_x)
            max_x = numpy.max(array_x)
            line.slope, line.intercept = YoonLine2D._list_square(array_x, array_y)
            line.start_pos = YoonVector2D(min_x, line.y(min_x))
            line.end_pos = YoonVector2D(max_x, line.y(max_x))
        return line

    @classmethod
    def from_vectors(cls,
                     *args: YoonVector2D):
        line = YoonLine2D()
        if len(args) > 0:
            array_x = YoonVector2D.vectors_to_array_x(args)
            array_y = YoonVector2D.vectors_to_array_y(args)
            min_x = YoonVector2D.minimum_x_in_vectors(args)
            max_x = YoonVector2D.maximum_x_in_vectors(args)
            line.slope, line.intercept = YoonLine2D._list_square(array_x, array_y)
            line.start_pos = YoonVector2D(min_x, line.y(min_x))
            line.end_pos = YoonVector2D(max_x, line.y(max_x))
        return line

    @classmethod
    def from_pos(cls,
                 start_vector: YoonVector2D = None,
                 end_vector: YoonVector2D = None):
        line = YoonLine2D()
        line.slope = (start_vector.y - end_vector.y) / (start_vector.x - end_vector.x)
        line.intercept = start_vector.y - line.slope * start_vector.x
        line.start_pos = start_vector.__copy__()
        line.end_pos = end_vector.__copy__()
        return line

    def __init__(self,
                 slope=0.0,
                 intercept=0.0):
        self.slope = slope
        self.intercept = intercept
        self.start_pos = YoonVector2D(0, self.y(0))
        self.end_pos = YoonVector2D(100, self.y(100))

    def __str__(self):
        return "SLOPE : {0}, INTERCEPT : {1}".format(self.slope, self.intercept)

    def __copy__(self):
        return YoonLine2D.from_pos(self.start_pos, self.end_pos)

    def x(self, y: (int, float)):
        assert isinstance(y, (int, float))
        return (y - self.intercept) / self.slope

    def y(self, x: (int, float)):
        assert isinstance(x, (int, float))
        return x * self.slope + self.intercept

    def feature_pos(self):
        return (self.start_pos + self.end_pos) / 2

    def distance(self, pos: YoonVector2D):
        assert isinstance(pos, YoonVector2D)
        return abs(self.slope * pos.x - pos.y + self.intercept) / math.sqrt(self.slope ** 2 + 1)

    def is_contain(self, pos: YoonVector2D):
        assert isinstance(pos, YoonVector2D)
        return pos.y == pos.x * self.slope + self.intercept

    def __add__(self, other):
        assert isinstance(other, YoonLine2D)
        return YoonLine2D(slope=self.slope + other.slope, intercept=self.intercept + other.intercept)

    def __sub__(self, other):
        assert isinstance(other, YoonLine2D)
        return YoonLine2D(slope=self.slope - other.slope, intercept=self.intercept + other.intercept)

    def __eq__(self, other):
        assert isinstance(other, YoonLine2D)
        return (self.slope == other.slope) and (self.intercept == other.intercept)


class YoonRect2D:
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    """

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 width=0.0,
                 height=0.0):
        self.center_pos = YoonVector2D(x, y).__copy__()
        self.width = width
        self.height = height

    @classmethod
    def from_list(cls, args: list):
        rect = YoonRect2D()
        if args is not None:
            min_x = YoonVector2D.minimum_x_in_list(args)
            min_y = YoonVector2D.minimum_y_in_list(args)
            max_x = YoonVector2D.maximum_x_in_list(args)
            max_y = YoonVector2D.maximum_y_in_list(args)
            rect.center_pos.x = (min_x + max_x) / 2
            rect.center_pos.y = (min_y + max_y) / 2
            rect.width = max_x - min_x
            rect.height = max_y - min_y
        return rect

    @classmethod
    def from_vectors(cls, *args: YoonVector2D):
        rect = YoonRect2D()
        if len(args) > 0:
            min_x = YoonVector2D.minimum_x_in_vectors(args)
            min_y = YoonVector2D.minimum_y_in_vectors(args)
            max_x = YoonVector2D.maximum_x_in_vectors(args)
            max_y = YoonVector2D.maximum_y_in_vectors(args)
            rect.center_pos.x = (min_x + max_x) / 2
            rect.center_pos.y = (min_y + max_y) / 2
            rect.width = max_x - min_x
            rect.height = max_y - min_y
        return rect

    @classmethod
    def from_array(cls, array: ndarray):  # top, left, bottom, right
        assert len(array.shape) == 1 and array.shape[0] == 4
        x = (array[1] + array[3]) / 2
        y = (array[0] + array[2]) / 2
        width = abs(array[3] - array[1])
        height = abs(array[2] - array[0])
        return YoonRect2D(x, y, width, height)

    @classmethod
    def from_dir_pair(cls, **kwargs):
        rect = YoonRect2D()
        if kwargs.get("dir1") and kwargs.get("dir2") and kwargs.get("pos1") and kwargs.get("pos2"):
            assert isinstance(kwargs["dir1"], YoonDir2D)
            assert isinstance(kwargs["dir2"], YoonDir2D)
            dir1 = kwargs["dir1"]
            dir2 = kwargs["dir2"]
            pos1 = kwargs["pos1"]
            pos2 = kwargs["pos2"]
            if dir1 == YoonDir2D.TOP_LEFT and dir2 == YoonDir2D.BOTTOM_RIGHT:
                rect.center_pos = (pos1 + pos2) / 2
                rect.width = (pos2 - pos1).x
                rect.height = (pos2 - pos1).y
            elif dir1 == YoonDir2D.BOTTOM_RIGHT and dir2 == YoonDir2D.TOP_LEFT:
                rect.center_pos = (pos1 + pos2) / 2
                rect.width = (pos1 - pos2).x
                rect.height = (pos1 - pos2).y
            elif dir1 == YoonDir2D.TOP_RIGHT and dir2 == YoonDir2D.BOTTOM_RIGHT:
                rect.center_pos = (pos1 + pos2) / 2
                rect.width = (pos2 - pos1).x
                rect.height = (pos1 - pos2).y
            elif dir1 == YoonDir2D.BOTTOM_LEFT and dir2 == YoonDir2D.TOP_RIGHT:
                rect.center_pos = (pos1 + pos2) / 2
                rect.width = (pos1 - pos2).x
                rect.height = (pos2 - pos1).y
        return rect

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, CENTER {2}, TL : {3}, BR : {4}".format(self.width, self.height,
                                                                                  self.center_pos.__str__(),
                                                                                  self.top_left().__str__(),
                                                                                  self.bottom_right().__str__())

    def __copy__(self):
        return YoonRect2D(x=self.center_pos.x, y=self.center_pos.y, width=self.width, height=self.height)

    def left(self):
        return self.center_pos.x - self.width / 2

    def top(self):
        return self.center_pos.y - self.height / 2

    def right(self):
        return self.center_pos.x + self.width / 2

    def bottom(self):
        return self.center_pos.y + self.height / 2

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
        list_x = [self.top_right().x, self.top_left().x, self.bottom_left().x, self.bottom_right().x]
        list_y = [self.top_right().y, self.top_left().y, self.bottom_left().y, self.bottom_right().y]
        return numpy.array(list_x, list_y)

    def to_list(self):
        return [self.left(), self.top(), self.width, self.height]

    def to_tuple(self):
        return self.left(), self.top(), self.width, self.height

    def area(self):
        return self.width * self.height

    def is_contain(self, pos: YoonVector2D):
        assert isinstance(pos, YoonVector2D)
        if self.left() < pos.x < self.right() and self.top() < pos.y < self.bottom():
            return True
        else:
            return False

    def feature_pos(self):
        return (self.top_left() + self.bottom_right()) / 2

    def __add__(self, other):
        assert isinstance(other, YoonRect2D)
        top = min(self.top(), other.top(), self.bottom(), other.bottom())
        bottom = max(self.top(), other.top(), self.bottom(), other.bottom())
        left = min(self.left(), other.left(), self.right(), other.right())
        right = max(self.left(), other.left(), self.right(), other.right())
        return YoonRect2D(x=(left + right) / 2, y=(top + bottom) / 2, width=right - left, height=bottom - top)

    def __mul__(self, scale: (int, float)):
        assert isinstance(scale, (int, float))
        return YoonRect2D(x=self.center_pos.x, y=self.center_pos.y,
                          width=scale * self.width, height=scale * self.height)

    def __eq__(self, other: (int, float)):
        assert isinstance(other, YoonRect2D)
        return self.center_pos == other.center_pos and self.width == other.width and self.height == other.height
