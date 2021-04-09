import cv2
import numpy

from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D


class YoonImage:
    width = 640
    height = 480
    bpp = 1
    __buffer = numpy.zeros((height, width, bpp), dtype=numpy.uint8)

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, PLANE : {2}".format(self.width, self.height, self.bpp)

    def __init__(self, **kwargs):
        if kwargs.get("buffer") is not None:
            assert isinstance(kwargs.get("buffer"), numpy.ndarray)
            self.__buffer = kwargs.get("buffer")
            self.height, self.width, self.bpp = self.__buffer.shape
        elif kwargs.get("image") is not None:
            assert isinstance(kwargs.get("image"), YoonImage)
            self.__buffer = kwargs.get("image").copy_buffer()
            self.width = kwargs.get("image").width
            self.height = kwargs.get("image").height
            self.bpp = kwargs.get("image").bpp
        elif kwargs.get("file"):
            assert isinstance(kwargs.get("file"), str)
            self.__buffer = cv2.imread(kwargs.get("file"))
            self.height, self.width, self.bpp = self.__buffer.shape
        else:
            if kwargs.get("width"):
                self.width = kwargs.get("width")
            if kwargs.get("height"):
                self.height = kwargs.get("height")
            if kwargs.get("bpp"):
                self.bpp = kwargs.get("bpp")
            self.__buffer = numpy.zeros((self.height, self.width, self.bpp), dtype=numpy.uint8)

    def get_buffer(self):
        return self.__buffer

    def __copy__(self):
        return YoonImage(buffer=self.__buffer)

    def copy_buffer(self):
        return self.__buffer.copy()

    def crop(self, pRect: YoonRect2D):
        assert isinstance(pRect, YoonRect2D)
        return YoonImage(buffer=self.__buffer[int(pRect.top()): int(pRect.bottom()), int(pRect.left()): int(pRect.right())].copy())

    def resize(self, scaleX: (int, float), scaleY: (int, float)):
        pResultBuffer = cv2.resize(self.__buffer, None, fx=scaleX, fy=scaleY)
        return YoonImage(buffer=pResultBuffer)

    def draw_rectangle(self, pRect: YoonRect2D, arrayColor: numpy.ndarray):
        cv2.rectangle(self.__buffer, (int(pRect.left()), int(pRect.top())), (int(pRect.right()), int(pRect.bottom())), arrayColor, 2)

    def draw_text(self, strText: str, pVector: YoonVector2D, arrayColor: numpy.ndarray):
        cv2.putText(self.__buffer, strText, (int(pVector.x), int(pVector.y)), cv2.FONT_HERSHEY_PLAIN, 3, arrayColor, 3)

    def show_image(self):
        cv2.imshow("Image", self.__buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
