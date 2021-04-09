import math
from yoonpytory.vector import YoonVector2D
from yoonpytory.math import *


class YoonLine2D:
    slope = 0
    intercept = 0
    start_pos = YoonVector2D(0, 0)
    end_pos = YoonVector2D(0, 0)

    def __str__(self):
        return "SLOPE : {0}, INTERCEPT : {1}".format(self.slope, self.intercept)

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], YoonVector2D)
            arrayX = YoonVector2D.to_array_x(args)
            arrayY = YoonVector2D.to_array_y(args)
            minX = YoonVector2D.minimum_x(args)
            minY = YoonVector2D.maximum_x(args)
            self.slope, self.intercept = least_square(arrayX, arrayY)
            self.start_pos = YoonVector2D(minX, self.y(minX))
            self.end_pos = YoonVector2D(minY, self.y(minY))
        else:
            if kwargs.get("list"):
                for i in range(len(kwargs["list"])):
                    assert isinstance(kwargs["list"][i], YoonVector2D)
                arrayX = YoonVector2D.to_array_x(kwargs["list"])
                arrayY = YoonVector2D.to_array_y(kwargs["list"])
                self.slope, self.intercept = least_square(arrayX, arrayY)
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
                minX = kwargs["x1"] if kwargs["x1"] < kwargs["x2"] else kwargs["x2"]
                minY = kwargs["x1"] if kwargs["x1"] > kwargs["x2"] else kwargs["x2"]
                self.start_pos = YoonVector2D(minX, self.y(minX))
                self.end_pos = YoonVector2D(minY, self.y(minY))

    def __copy__(self):
        return YoonLine2D(slope=self.slope, intercept=self.intercept)

    def x(self, y: (int, float)):
        assert isinstance(y, (int, float))
        return (y - self.intercept) / self.slope

    def y(self, x: (int, float)):
        assert isinstance(x, (int, float))
        return x * self.slope + self.intercept

    def distance(self, pVector: YoonVector2D):
        assert isinstance(pVector, YoonVector2D)
        return abs(self.slope * pVector.x - pVector.y + self.intercept) / math.sqrt(self.slope ** 2 + 1)

    def is_contain(self, pVector: YoonVector2D):
        assert isinstance(pVector, YoonVector2D)
        return pVector.y == pVector.x * self.slope + self.intercept

    def __add__(self, pLine):
        assert isinstance(pLine, YoonLine2D)
        return YoonLine2D(slope=self.slope + pLine.slope, intercept=self.intercept + pLine.intercept)

    def __sub__(self, pLine):
        assert isinstance(pLine, YoonLine2D)
        return YoonLine2D(slope=self.slope - pLine.slope, intercept=self.intercept + pLine.intercept)

    def __eq__(self, pLine):
        assert isinstance(pLine, YoonLine2D)
        return (self.slope == pLine.slope) and (self.intercept == pLine.intercept)
