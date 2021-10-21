from enum import Enum


class YoonDir2D(Enum):
    NONE = -1
    CENTER = 0
    TOP_LEFT = 1
    TOP = 2
    TOP_RIGHT = 3
    RIGHT = 4
    BOTTOM_RIGHT = 5
    BOTTOM = 6
    BOTTOM_LEFT = 7
    LEFT = 8

    def __str__(self):
        return "{0}".format(self.name)

    def go(self, tag: str):
        dic = {"clock4": self.previous_quadrant(),
               "anticlock4": self.next_quadrant(),
               "clock8": self.previous_octant(),
               "anticlock8": self.next_octant(),
               "order": self.next_order(),
               "antiorder": self.previous_order(),
               "whirlpool": self.next_whirlpool(),
               "antiwhirlpool": self.previous_whirlpool(),
               "x": self.reverse_y(),
               "y": self.reverse_x()}
        if dic.get(tag):
            return dic[tag]
        else:
            return self

    def back(self, tag: str):
        dic = {"clock4": self.next_quadrant(),
               "anticlock4": self.previous_quadrant(),
               "clock8": self.next_octant(),
               "anticlock8": self.previous_octant(),
               "order": self.previous_order(),
               "antiorder": self.next_order(),
               "whirlpool": self.previous_whirlpool(),
               "antiwhirlpool": self.next_whirlpool(),
               "x": self.reverse_y(),
               "y": self.reverse_x()}
        if dic.get(tag):
            return dic[tag]
        else:
            return self

    def next_quadrant(self):
        dic = {"TOP": self.LEFT,
               "LEFT": self.BOTTOM,
               "BOTTOM": self.RIGHT,
               "RIGHT": self.TOP,
               "TOP_RIGHT": self.TOP_LEFT,
               "TOP_LEFT": self.BOTTOM_LEFT,
               "BOTTOM_LEFT": self.BOTTOM_RIGHT,
               "BOTTOM_RIGHT": self.TOP_RIGHT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def previous_quadrant(self):
        dic = {"TOP": self.RIGHT,
               "RIGHT": self.BOTTOM,
               "BOTTOM": self.LEFT,
               "LEFT": self.TOP,
               "TOP_RIGHT": self.BOTTOM_RIGHT,
               "BOTTOM_RIGHT": self.BOTTOM_LEFT,
               "BOTTOM_LEFT": self.TOP_LEFT,
               "TOP_LEFT": self.TOP_RIGHT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def next_octant(self):
        dic = {"RIGHT": self.TOP_RIGHT,
               "TOP_RIGHT": self.TOP,
               "TOP": self.TOP_LEFT,
               "TOP_LEFT": self.LEFT,
               "LEFT": self.BOTTOM_LEFT,
               "BOTTOM_LEFT": self.BOTTOM,
               "BOTTOM": self.BOTTOM_RIGHT,
               "BOTTOM_RIGHT": self.RIGHT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def previous_octant(self):
        dic = {"RIGHT": self.BOTTOM_RIGHT,
               "BOTTOM_RIGHT": self.BOTTOM,
               "BOTTOM": self.BOTTOM_LEFT,
               "BOTTOM_LEFT": self.LEFT,
               "LEFT": self.TOP_LEFT,
               "TOP_LEFT": self.TOP,
               "TOP": self.TOP_RIGHT,
               "TOP_RIGHT": self.RIGHT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def reverse_x(self):
        dic = {"TOP_LEFT": self.BOTTOM_LEFT,
               "TOP": self.BOTTOM,
               "TOP_RIGHT": self.BOTTOM_RIGHT,
               "BOTTOM_LEFT": self.TOP_LEFT,
               "BOTTOM": self.TOP,
               "BOTTOM_RIGHT": self.TOP_RIGHT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def reverse_y(self):
        dic = {"TOP_LEFT": self.TOP_RIGHT,
               "LEFT": self.RIGHT,
               "BOTTOM_LEFT": self.BOTTOM_RIGHT,
               "TOP_RIGHT": self.TOP_LEFT,
               "RIGHT": self.LEFT,
               "BOTTOM_RIGHT": self.BOTTOM_LEFT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def next_order(self):
        dic = {"TOP_LEFT": self.TOP,
               "TOP": self.TOP_RIGHT,
               "TOP_RIGHT": self.LEFT,
               "LEFT": self.CENTER,
               "CENTER": self.RIGHT,
               "RIGHT": self.BOTTOM_LEFT,
               "BOTTOM_LEFT": self.BOTTOM,
               "BOTTOM": self.BOTTOM_RIGHT,
               "BOTTOM_RIGHT": self.TOP_LEFT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def previous_order(self):
        dic = {"BOTTOM_RIGHT": self.BOTTOM,
               "BOTTOM": self.BOTTOM_LEFT,
               "BOTTOM_LEFT": self.RIGHT,
               "RIGHT": self.CENTER,
               "CENTER": self.LEFT,
               "LEFT": self.TOP_RIGHT,
               "TOP_RIGHT": self.TOP,
               "TOP": self.TOP_LEFT,
               "TOP_LEFT": self.BOTTOM_RIGHT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def next_whirlpool(self):
        dic = {"TOP_LEFT": self.TOP,
               "TOP": self.TOP_RIGHT,
               "TOP_RIGHT": self.RIGHT,
               "RIGHT": self.BOTTOM_RIGHT,
               "BOTTOM_RIGHT": self.BOTTOM,
               "BOTTOM": self.BOTTOM_LEFT,
               "BOTTOM_LEFT": self.LEFT,
               "LEFT": self.CENTER,
               "CENTER": self.TOP_LEFT}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def previous_whirlpool(self):
        dic = {"CENTER": self.LEFT,
               "LEFT": self.BOTTOM_LEFT,
               "BOTTOM_LEFT": self.BOTTOM,
               "BOTTOM": self.BOTTOM_RIGHT,
               "BOTTOM_RIGHT": self.RIGHT,
               "RIGHT": self.TOP_RIGHT,
               "TOP_RIGHT": self.TOP,
               "TOP": self.TOP_LEFT,
               "TOP_LEFT": self.CENTER}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return self

    def to_order_number(self):
        dic = {"TOP_LEFT": 0,
               "TOP": 1,
               "TOP_RIGHT": 2,
               "LEFT": 3,
               "CENTER": 4,
               "RIGHT": 5,
               "BOTTOM_LEFT": 6,
               "BOTTOM": 7,
               "BOTTOM_RIGHT": 8}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return -1

    @classmethod
    def from_order_num(cls, order):
        dic = {0: cls.TOP_LEFT,
               1: cls.TOP,
               2: cls.TOP_RIGHT,
               3: cls.LEFT,
               4: cls.CENTER,
               5: cls.RIGHT,
               6: cls.BOTTOM_LEFT,
               7: cls.BOTTOM,
               8: cls.BOTTOM_RIGHT}
        if dic.get(order):
            return dic[order]
        else:
            return cls.NONE

    def to_clock_number(self):
        dic = {"TOP": 0,
               "TOP_RIGHT": 1,
               "RIGHT": 3,
               "RIGHT_BOTTOM": 5,
               "BOTTOM": 6,
               "BOTTOM_LEFT": 7,
               "LEFT": 9,
               "TOP_LEFT": 11}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return -1

    @classmethod
    def from_clock_number(cls, clock: int):
        if clock in [0, 12]:
            return cls.TOP
        elif clock in [1, 2]:
            return cls.TOP_RIGHT
        elif clock == 3:
            return cls.RIGHT
        elif clock in [4, 5]:
            return cls.BOTTOM_RIGHT
        elif clock == 6:
            return cls.BOTTOM
        elif clock in [7, 8]:
            return cls.BOTTOM_LEFT
        elif clock == 9:
            return cls.LEFT
        elif clock in [10, 11]:
            return cls.TOP_LEFT
        else:
            raise cls.NONE

    def to_quadrant(self):
        dic = {"CENTER": 0,
               "RIGHT": 1,
               "TOP_RIGHT": 1,
               "TOP": 2,
               "TOP_LEFT": 2,
               "LEFT": 3,
               "BOTTOM_LEFT": 3,
               "BOTTOM": 4,
               "BOTTOM_RIGHT": 4}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return -1

    @classmethod
    def from_quadrant(cls, quad: int):
        dic = {0: cls.CENTER,
               1: cls.TOP_RIGHT,
               2: cls.TOP_LEFT,
               3: cls.BOTTOM_LEFT,
               4: cls.BOTTOM_RIGHT, }
        if dic.get(quad):
            return dic[quad]
        else:
            return cls.NONE

    def to_tuple(self):
        dic = {"CENTER": (0, 0),
               "RIGHT": (1, 0),
               "TOP_RIGHT": (1, 1),
               "TOP": (0, 1),
               "TOP_LEFT": (1, -1),
               "LEFT": (-1, 0),
               "BOTTOM_LEFT": (-1, -1),
               "BOTTOM": (0, -1),
               "BOTTOM_RIGHT": (-1, 1)}
        if dic.get(self.name):
            return dic[self.name]
        else:
            return 0, 0

    @classmethod
    def from_tuple(cls, pair: tuple):
        assert isinstance(pair, tuple)
        ndX = pair[0]
        ndY = pair[1]
        if ndX == 0 and ndY == 0:
            return YoonDir2D.CENTER
        elif ndX == 0 and ndY > 0:
            return YoonDir2D.TOP
        elif ndX == 0 and ndY < 0:
            return YoonDir2D.BOTTOM
        elif ndX > 0 and ndY == 0:
            return YoonDir2D.RIGHT
        elif ndX < 0 and ndY == 0:
            return YoonDir2D.LEFT
        elif ndX > 0 and ndY > 0:
            return YoonDir2D.TOP_RIGHT
        elif ndX < 0 < ndY:
            return YoonDir2D.TOP_LEFT
        elif ndX < 0 and ndY < 0:
            return YoonDir2D.BOTTOM_LEFT
        elif ndX > 0 > ndY:
            return YoonDir2D.BOTTOM_RIGHT
        else:
            return YoonDir2D.NONE

    @classmethod
    def get_clock_directions(cls):
        return [cls.RIGHT, cls.TOP_RIGHT, cls.TOP, cls.TOP_LEFT, cls.LEFT, cls.BOTTOM_LEFT, cls.BOTTOM,
                cls.BOTTOM_RIGHT]

    @classmethod
    def get_square_directions(cls):
        return [cls.TOP_RIGHT, cls.TOP_LEFT, cls.BOTTOM_LEFT, cls.BOTTOM_RIGHT]

    @classmethod
    def get_rhombus_directions(cls):
        return [cls.TOP, cls.LEFT, cls.BOTTOM, cls.RIGHT]

    @classmethod
    def get_horizon_directions(cls):
        return [cls.LEFT, cls.RIGHT]

    @classmethod
    def get_vertical_directions(cls):
        return [cls.TOP, cls.BOTTOM]

    def __add__(self, other):
        assert isinstance(other, (YoonDir2D, str))
        if isinstance(other, YoonDir2D):
            originX, originY = self.to_tuple()
            inputX, inputY = other.to_tuple()
            return YoonDir2D.from_tuple((originX + inputX, originY + inputY))
        else:
            return self.go(other)

    def __sub__(self, other):
        assert isinstance(other, (YoonDir2D, str))
        if isinstance(other, YoonDir2D):
            originX, originY = self.to_tuple()
            inputX, inputY = other.to_tuple()
            return YoonDir2D.from_tuple((originX - inputX, originY - inputY))
        else:
            return self.back(other)

    def __eq__(self, other):
        assert isinstance(other, YoonDir2D)
        return self.name == other.name
