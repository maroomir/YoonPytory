import math
from yoonpytory.vector import YoonVector2D
from yoonpytory.math import *


class YoonLine2D:
    slope = 0
    intercept = 0
    start_pos = YoonVector2D()
    end_pos = YoonVector2D()

    def __str__(self):
        return "SLOPE : {0}, INTERCEPT : {1}".format(self.slope, self.intercept)

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
            array_x = YoonVector2D.to_ndarray_x(args)
            array_y = YoonVector2D.to_ndarray_y(args)
            min_x = YoonVector2D.minimum_x(args)
            max_x = YoonVector2D.maximum_x(args)
            self.slope, self.intercept = least_square(array_x, array_y)
            self.start_pos = YoonVector2D(min_x, self.y(min_x))
            self.end_pos = YoonVector2D(max_x, self.y(max_x))
        else:
            if kwargs.get("list"):
                for i in range(len(kwargs["list"])):
                    assert isinstance(kwargs["list"][i], YoonVector2D)
                array_x = YoonVector2D.to_ndarray_x(kwargs["list"])
                array_y = YoonVector2D.to_ndarray_y(kwargs["list"])
                self.slope, self.intercept = least_square(array_x, array_y)
            elif kwargs.get("slope"):
                assert isinstance(kwargs["slope"], (int, float))
                self.slope = kwargs["slope"]
            elif kwargs.get("intercept"):
                assert isinstance(kwargs["intercept"], (int, float))
                self.intercept = kwargs["intercept"]
            elif kwargs.get("x1") and kwargs.get("x2") and kwargs.get("y1") and kwargs.get("y2"):
                assert isinstance(kwargs["x1"], (int, float))
                assert isinstance(kwargs["x2"], (int, float))
                assert isinstance(kwargs["y1"], (int, float))
                assert isinstance(kwargs["y2"], (int, float))
                self.slope = (kwargs["y1"] - kwargs["y2"]) / (kwargs["x1"] - kwargs["x2"])
                self.intercept = kwargs["y1"] - self.slope * kwargs["x1"]
                min_x = kwargs["x1"] if kwargs["x1"] < kwargs["x2"] else kwargs["x2"]
                max_x = kwargs["x1"] if kwargs["x1"] > kwargs["x2"] else kwargs["x2"]
                self.start_pos = YoonVector2D(min_x, self.y(min_x))
                self.end_pos = YoonVector2D(max_x, self.y(max_x))

    def __copy__(self):
        return YoonLine2D(slope=self.slope, intercept=self.intercept)

    def x(self, y):
        assert isinstance(y, (int, float))
        return (y - self.intercept) / self.slope

    def y(self, x):
        assert isinstance(x, (int, float))
        return x * self.slope + self.intercept

    def distance(self, vector):
        assert isinstance(vector, YoonVector2D)
        return abs(self.slope * vector.x - vector.y + self.intercept) / math.sqrt(self.slope ** 2 + 1)

    def is_contain(self, vector):
        assert isinstance(vector, YoonVector2D)
        return vector.y == vector.x * self.slope + self.intercept

    def __add__(self, other):
        assert isinstance(other, YoonLine2D)
        return YoonLine2D(slope=self.slope + other.slope, intercept=self.intercept + other.intercept)

    def __sub__(self, other):
        assert isinstance(other, YoonLine2D)
        return YoonLine2D(slope=self.slope - other.slope, intercept=self.intercept + other.intercept)