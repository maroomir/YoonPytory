import sys
import numpy
from yoonpytory.dir import eYoonDir2D
from yoonpytory.vector import YoonVector2D


class YoonRect2D:
    width = 0
    height = 0
    center_pos = YoonVector2D(0, 0)

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, CENTER {2}, TL : {3}, BR : {4}".format(self.width, self.height, self.center_pos.__str__(), self.top_left().__str__(), self.bottom_right().__str__())

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            min_x = min_y = sys.maxsize
            max_x = max_y = -sys.maxsize
            for vector in args:
                assert isinstance(vector, YoonVector2D)
                if vector.x < min_x:
                    min_x = vector.x
                elif vector.x > max_x:
                    max_x = vector.x
                if vector.y < min_y:
                    min_y = vector.y
                elif vector.y > max_y:
                    max_y = vector.y
            self.center_pos.x = (min_x + max_x) / 2
            self.center_pos.y = (min_y + max_y) / 2
            self.width = max_x - min_x
            self.height = max_y - min_y
        else:
            if kwargs.get("list"):
                assert isinstance(kwargs["list"], list)
                min_x = min_y = sys.maxsize
                max_x = max_y = -sys.maxsize
                for vector in kwargs["list"]:
                    assert isinstance(vector, YoonVector2D)
                    if vector.x < min_x:
                        min_x = vector.x
                    elif vector.x > max_x:
                        max_x = vector.x
                    if vector.y < min_y:
                        min_y = vector.y
                    elif vector.y > max_y:
                        max_y = vector.y
                self.center_pos.x = (min_x + max_x) / 2
                self.center_pos.y = (min_y + max_y) / 2
                self.width = max_x - min_x
                self.height = max_y - min_y
            elif kwargs.get("dir1") and kwargs.get("dir2") and kwargs.get("pos1") and kwargs.get("pos2"):
                assert isinstance(kwargs["dir1"], eYoonDir2D)
                assert isinstance(kwargs["dir2"], eYoonDir2D)
                dir1 = kwargs["dir1"]
                dir2 = kwargs["dir2"]
                pos1 = kwargs["pos1"]
                pos2 = kwargs["pos2"]
                if dir1 == eYoonDir2D.TOP_LEFT and dir2 == eYoonDir2D.BOTTOM_RIGHT:
                    self.center_pos = (pos1 + pos2) / 2
                    self.width = (pos2 - pos1).x
                    self.height = (pos2 - pos1).y
                elif dir1 == eYoonDir2D.BOTTOM_RIGHT and dir2 == eYoonDir2D.TOP_LEFT:
                    self.center_pos = (pos1 + pos2) / 2
                    self.width = (pos1 - pos2).x
                    self.height = (pos1 - pos2).y
                elif dir1 == eYoonDir2D.TOP_RIGHT and dir2 == eYoonDir2D.BOTTOM_RIGHT:
                    self.center_pos = (pos1 + pos2) / 2
                    self.width = (pos2 - pos1).x
                    self.height = (pos1 - pos2).y
                elif dir1 == eYoonDir2D.BOTTOM_LEFT and dir2 == eYoonDir2D.TOP_RIGHT:
                    self.center_pos = (pos1 + pos2) / 2
                    self.width = (pos1 - pos2).x
                    self.height = (pos2 - pos1).y
            else:
                input_x = 0
                input_y = 0
                input_width = 0
                input_height = 0
                if kwargs.get("pos"):
                    assert isinstance(kwargs["pos"], YoonVector2D)
                    input_x = kwargs["pos"].x
                    input_y = kwargs["pos"].y
                if kwargs.get("x"):
                    assert isinstance(kwargs["x"], (int, float))
                    input_x = kwargs["x"]
                if kwargs.get("y"):
                    assert isinstance(kwargs["y"], (int, float))
                    input_y = kwargs["y"]
                if kwargs.get("width"):
                    assert isinstance(kwargs["width"], (int, float))
                    input_width = kwargs["width"]
                if kwargs.get("height"):
                    assert isinstance(kwargs["height"], (int, float))
                    input_height = kwargs["height"]
                self.center_pos = YoonVector2D(input_x, input_y)
                self.width = input_width
                self.height = input_height

    def __copy__(self):
        return YoonRect2D(pos=self.center_pos, width=self.width, height=self.height)

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
        listX = [self.top_right().x, self.top_left().x, self.bottom_left().x, self.bottom_right().x]
        listY = [self.top_right().y, self.top_left().y, self.bottom_left().y, self.bottom_right().y]
        return numpy.array(listX, listY)

    def to_list(self):
        return [self.left(), self.top(), self.width, self.height]

    def to_tuple(self):
        return self.left(), self.top(), self.width, self.height

    def area(self):
        return self.width * self.height

    def is_contain(self, pVector: YoonVector2D):
        assert isinstance(pVector, YoonVector2D)
        if self.left() < pVector.x < self.right() and self.top() < pVector.y < self.bottom():
            return True
        else:
            return False

    def __add__(self, pRect):
        assert isinstance(pRect, YoonRect2D)
        top = min(self.top(), pRect.top(), self.bottom(), pRect.bottom())
        bottom = max(self.top(), pRect.top(), self.bottom(), pRect.bottom())
        left = min(self.left(), pRect.left(), self.right(), pRect.right())
        right = max(self.left(), pRect.left(), self.right(), pRect.right())
        return YoonRect2D(x=(left + right) / 2, y=(top + bottom) / 2, width=right - left, height=bottom - top)

    def __mul__(self, scale: (int, float)):
        assert isinstance(scale, (int, float))
        return YoonRect2D(pos=self.center_pos.__copy__(), width=scale * self.width, height=scale * self.height)

    def __eq__(self, pRect: (int, float)):
        assert isinstance(pRect, YoonRect2D)
        return self.center_pos == pRect.center_pos and self.width == pRect.width and self.height == pRect.height
