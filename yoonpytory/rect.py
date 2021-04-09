import sys
import numpy
from yoonpytory.dir import eYoonDir2D
from yoonpytory.vector import YoonVector2D


class YoonRect2D:
    width: (int, float)
    height: (int, float)
    centerPos = YoonVector2D(0, 0)

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, CENTER {2}, TL : {3}, BR : {4}".format(self.width, self.height,
                                                                                  self.centerPos.__str__(),
                                                                                  self.top_left().__str__(),
                                                                                  self.bottom_right().__str__())

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
                    self.centerPos = YoonVector2D(dX, dY)
                self.width = dWidth
                self.height = dHeight

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
