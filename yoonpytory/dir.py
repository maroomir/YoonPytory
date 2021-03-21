from enum import Enum


class eYoonDir2D(Enum):
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

    def go(self, strTag):
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
        if dic.get(strTag):
            return dic[strTag]
        else:
            return self

    def back(self, strTag):
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
        if dic.get(strTag):
            return dic[strTag]
        else:
            return self

    def next_quadrant(self):
        dic = {self.TOP: self.LEFT,
               self.LEFT: self.BOTTOM,
               self.BOTTOM: self.RIGHT,
               self.RIGHT: self.TOP,
               self.TOP_RIGHT: self.TOP_LEFT,
               self.TOP_LEFT: self.BOTTOM_LEFT,
               self.BOTTOM_LEFT: self.BOTTOM_RIGHT,
               self.BOTTOM_RIGHT: self.TOP_RIGHT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def previous_quadrant(self):
        dic = {self.TOP: self.RIGHT,
               self.RIGHT: self.BOTTOM,
               self.BOTTOM: self.LEFT,
               self.LEFT: self.TOP,
               self.TOP_RIGHT: self.BOTTOM_RIGHT,
               self.BOTTOM_RIGHT: self.BOTTOM_LEFT,
               self.BOTTOM_LEFT: self.TOP_LEFT,
               self.TOP_LEFT: self.TOP_RIGHT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def next_octant(self):
        dic = {self.RIGHT: self.TOP_RIGHT,
               self.TOP_RIGHT: self.TOP,
               self.TOP: self.TOP_LEFT,
               self.TOP_LEFT: self.LEFT,
               self.LEFT: self.BOTTOM_LEFT,
               self.BOTTOM_LEFT: self.BOTTOM,
               self.BOTTOM: self.BOTTOM_RIGHT,
               self.BOTTOM_RIGHT: self.RIGHT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def previous_octant(self):
        dic = {self.RIGHT: self.BOTTOM_RIGHT,
               self.BOTTOM_RIGHT: self.BOTTOM,
               self.BOTTOM: self.BOTTOM_LEFT,
               self.BOTTOM_LEFT: self.LEFT,
               self.LEFT: self.TOP_LEFT,
               self.TOP_LEFT: self.TOP,
               self.TOP: self.TOP_RIGHT,
               self.TOP_RIGHT: self.RIGHT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def reverse_x(self):
        dic = {self.TOP_LEFT: self.BOTTOM_LEFT,
               self.TOP: self.BOTTOM,
               self.TOP_RIGHT: self.BOTTOM_RIGHT,
               self.BOTTOM_LEFT: self.TOP_LEFT,
               self.BOTTOM: self.TOP,
               self.BOTTOM_RIGHT: self.TOP_RIGHT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def reverse_y(self):
        dic = {self.TOP_LEFT: self.TOP_RIGHT,
               self.LEFT: self.RIGHT,
               self.BOTTOM_LEFT: self.BOTTOM_RIGHT,
               self.TOP_RIGHT: self.TOP_LEFT,
               self.RIGHT: self.LEFT,
               self.BOTTOM_RIGHT: self.BOTTOM_LEFT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def next_order(self):
        dic = {self.TOP_LEFT: self.TOP,
               self.TOP: self.TOP_RIGHT,
               self.TOP_RIGHT: self.LEFT,
               self.LEFT: self.CENTER,
               self.CENTER: self.RIGHT,
               self.RIGHT: self.BOTTOM_LEFT,
               self.BOTTOM_LEFT: self.BOTTOM,
               self.BOTTOM: self.BOTTOM_RIGHT,
               self.BOTTOM_RIGHT: self.TOP_LEFT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def previous_order(self):
        dic = {self.BOTTOM_RIGHT: self.BOTTOM,
               self.BOTTOM: self.BOTTOM_LEFT,
               self.BOTTOM_LEFT: self.RIGHT,
               self.RIGHT: self.CENTER,
               self.CENTER: self.LEFT,
               self.LEFT: self.TOP_RIGHT,
               self.TOP_RIGHT: self.TOP,
               self.TOP: self.TOP_LEFT,
               self.TOP_LEFT: self.BOTTOM_RIGHT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def next_whirlpool(self):
        dic = {self.TOP_LEFT: self.TOP,
               self.TOP: self.TOP_RIGHT,
               self.TOP_RIGHT: self.RIGHT,
               self.RIGHT: self.BOTTOM_RIGHT,
               self.BOTTOM_RIGHT: self.BOTTOM,
               self.BOTTOM: self.BOTTOM_LEFT,
               self.BOTTOM_LEFT: self.LEFT,
               self.LEFT: self.CENTER,
               self.CENTER: self.TOP_LEFT}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def previous_whirlpool(self):
        dic = {self.CENTER: self.LEFT,
               self.LEFT: self.BOTTOM_LEFT,
               self.BOTTOM_LEFT: self.BOTTOM,
               self.BOTTOM: self.BOTTOM_RIGHT,
               self.BOTTOM_RIGHT: self.RIGHT,
               self.RIGHT: self.TOP_RIGHT,
               self.TOP_RIGHT: self.TOP,
               self.TOP: self.TOP_LEFT,
               self.TOP_LEFT: self.CENTER}
        if dic.get(self):
            return dic[self]
        else:
            return self

    def to_order_number(self):
        dic = {self.TOP_LEFT: 0,
               self.TOP: 1,
               self.TOP_RIGHT: 2,
               self.LEFT: 3,
               self.CENTER: 4,
               self.RIGHT: 5,
               self.BOTTOM_LEFT: 6,
               self.BOTTOM: 7,
               self.BOTTOM_RIGHT: 8}
        if dic.get(self):
            return dic[self]
        else:
            return -1

    @classmethod
    def from_order_num(cls, nOrder):
        dic = {0: cls.TOP_LEFT,
               1: cls.TOP,
               2: cls.TOP_RIGHT,
               3: cls.LEFT,
               4: cls.CENTER,
               5: cls.RIGHT,
               6: cls.BOTTOM_LEFT,
               7: cls.BOTTOM,
               8: cls.BOTTOM_RIGHT}
        if dic.get(nOrder):
            return dic[nOrder]
        else:
            return cls.NONE

    def to_clock_number(self):
        dic = {self.TOP: 0,
               self.TOP_RIGHT: 1,
               self.RIGHT: 3,
               self.RIGHT_BOTTOM: 5,
               self.BOTTOM: 6,
               self.BOTTOM_LEFT: 7,
               self.LEFT: 9,
               self.TOP_LEFT: 11}
        if dic.get(self):
            return dic[self]
        else:
            return -1

    @classmethod
    def from_clock_number(cls, nClock):
        if nClock in [0, 12]:
            return cls.TOP
        elif nClock in [1, 2]:
            return cls.TOP_RIGHT
        elif nClock == 3:
            return cls.RIGHT
        elif nClock in [4, 5]:
            return cls.BOTTOM_RIGHT
        elif nClock == 6:
            return cls.BOTTOM
        elif nClock in [7, 8]:
            return cls.BOTTOM_LEFT
        elif nClock == 9:
            return cls.LEFT
        elif nClock in [10, 11]:
            return cls.TOP_LEFT
        else:
            raise cls.NONE

    def to_quadrant(self):
        dic = {self.CENTER: 0,
               self.RIGHT: 1,
               self.TOP_RIGHT: 1,
               self.TOP: 2,
               self.TOP_LEFT: 2,
               self.LEFT: 3,
               self.BOTTOM_LEFT: 3,
               self.BOTTOM: 4,
               self.BOTTOM_RIGHT: 4}
        if dic.get(self):
            return dic[self]
        else:
            return -1

    @classmethod
    def from_quadrant(cls, nQuad):
        dic = {0: cls.CENTER,
               1: cls.TOP_RIGHT,
               2: cls.TOP_LEFT,
               3: cls.BOTTOM_LEFT,
               4: cls.BOTTOM_RIGHT, }
        if dic.get(nQuad):
            return dic[nQuad]
        else:
            return cls.NONE

    def to_tuple(self):
        dic = {self.CENTER: (0, 0),
               self.RIGHT: (1, 0),
               self.TOP_RIGHT: (1, 1),
               self.TOP: (0, 1),
               self.TOP_LEFT: (1, -1),
               self.LEFT: (-1, 0),
               self.BOTTOM_LEFT: (-1, -1),
               self.BOTTOM: (0, -1),
               self.BOTTOM_RIGHT: (-1, 1)}
        if dic.get(self):
            return dic[self]
        else:
            return 0, 0

    @classmethod
    def from_tuple(cls, pPair):
        assert isinstance(pPair, tuple)
        x = pPair[0]
        y = pPair[1]
        if x == 0 and y == 0:
            return eYoonDir2D.CENTER
        elif x == 0 and y > 0:
            return eYoonDir2D.TOP
        elif x == 0 and y < 0:
            return eYoonDir2D.BOTTOM
        elif x > 0 and y == 0:
            return eYoonDir2D.RIGHT
        elif x < 0 and y == 0:
            return eYoonDir2D.LEFT
        elif x > 0 and y > 0:
            return eYoonDir2D.TOP_RIGHT
        elif x < 0 < y:
            return eYoonDir2D.TOP_LEFT
        elif x < 0 and y < 0:
            return eYoonDir2D.BOTTOM_LEFT
        elif x > 0 > y:
            return eYoonDir2D.BOTTOM_RIGHT
        else:
            return eYoonDir2D.NONE

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
        assert isinstance(other, (eYoonDir2D, str))
        if isinstance(other, eYoonDir2D):
            origin_x, origin_y = self.to_tuple()
            input_x, input_y = other.to_tuple()
            return eYoonDir2D.from_tuple((origin_x + input_x, origin_y + input_y))
        else:
            return self.go(other)

    def __sub__(self, other):
        assert isinstance(other, (eYoonDir2D, str))
        if isinstance(other, eYoonDir2D):
            origin_x, origin_y = self.to_tuple()
            input_x, input_y = other.to_tuple()
            return eYoonDir2D.from_tuple((origin_x - input_x, origin_y - input_y))
        else:
            return self.back(other)
