import numpy

from yoonpytory.dir import eYoonDir2D
from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D
from yoonimage.object import YoonObject
import cv2


class YoonImage:
    width = 640
    height = 480
    bpp = 1
    __buffer = numpy.zeros((height, width, bpp), dtype=numpy.uint8)

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, PLANE : {2}".format(self.width, self.height, self.bpp)

    def __init__(self, **kwargs):
        if kwargs.get("image"):
            assert isinstance(kwargs.get("image"), YoonImage)
            self.__buffer = kwargs.get("image").copy_buffer()
            self.width = kwargs.get("image").width
            self.height = kwargs.get("image").height
            self.bpp = kwargs.get("image").bpp
        elif kwargs.get("buffer"):
            assert isinstance(kwargs.get("buffer"), numpy.ndarray)
            self.__buffer = kwargs.get("buffer")
            self.height, self.width, self.bpp = self.__buffer.shape
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
        return YoonImage(image=self.__buffer)

    def copy_buffer(self):
        return self.__buffer.copy()

    def crop(self, rect: YoonRect2D):
        assert isinstance(rect, YoonRect2D)
        return YoonImage(buffer=self.__buffer[rect.top():rect.bottom(), rect.left():rect.right()])

    def search(self, imageTarget):
        assert isinstance(imageTarget, YoonImage)
        buffer_target = imageTarget.copy_buffer()
        vec_start = YoonVector2D()
        vec_end = YoonVector2D()
        for y in range(self.height - imageTarget.height):
            for x in range(self.width - imageTarget.width):
                if numpy.equal(self.__buffer[y:y + imageTarget.height, x:x + imageTarget.width], buffer_target):
                    vec_start = YoonVector2D(x, y)
                    vec_end = YoonVector2D(x + imageTarget.width, y + imageTarget.height)
                    break
        searched_rect = YoonRect2D(dir1=eYoonDir2D.TOP_LEFT, pos1=vec_start, dir2=eYoonDir2D.BOTTOM_RIGHT, pos2=vec_end)
        searched_image = self.crop(searched_rect)
        return YoonObject(score=100.0, object=searched_rect, image=searched_image)

    def resize(self, scale_x: (int, float), scale_y: (int, float)):
        buffer_result = cv2.resize(self.__buffer, None, fx=scale_x, fy=scale_y)
        return YoonImage(buffer=buffer_result)