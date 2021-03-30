import numpy

from yoonpytory.dir import eYoonDir2D
from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D


class YoonImage:
    width = 640
    height = 480
    __buffer = numpy.zeros((width, height), dtype=numpy.uint8)

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, LENGTH : {2}".format(self.width, self.height, self.__buffer.size)

    def __init__(self, **kwargs):
        if kwargs.get("image"):
            assert isinstance(kwargs.get("image"), YoonImage)
            self.__buffer = kwargs.get("image").copy_buffer()
            self.width = kwargs.get("image").width
            self.height = kwargs.get("image").height
        elif kwargs.get("buffer"):
            assert isinstance(kwargs.get("buffer"), numpy.ndarray)
            self.__buffer = kwargs.get("buffer")
            self.width, self.height = self.__buffer.shape
        else:
            if kwargs.get("width"):
                self.width = kwargs.get("width")
            if kwargs.get("height"):
                self.height = kwargs.get("height")
            self.__buffer = numpy.zeros((self.width, self.height), dtype=numpy.uint8)

    def __copy__(self):
        return YoonImage(image=self.__image)

    def copy_buffer(self):
        return self.__buffer.copy()

    def crop(self, rect):
        assert isinstance(rect, YoonRect2D)
        return YoonImage(width=rect.width, height=rect.height,
                         buffer=self.__buffer[rect.left():rect.right(), rect.top():rect.bottom()])

    def search(self, imageTarget):
        assert isinstance(imageTarget, YoonImage)
        buffer_target = imageTarget.copy_buffer()
        vec_start = YoonVector2D()
        vec_end = YoonVector2D()
        for y in range(self.height - imageTarget.height):
            for x in range(self.width - imageTarget.width):
                if numpy.equal(self.__buffer[x:x + imageTarget.width, y:y + imageTarget.height], buffer_target):
                    vec_start = YoonVector2D(x, y)
                    vec_end = YoonVector2D(x + imageTarget.width, y + imageTarget.height)
                    break
        return YoonRect2D(dir1=eYoonDir2D.TOP_LEFT, pos1=vec_start, dir2=eYoonDir2D.BOTTOM_RIGHT, pos2=vec_end)