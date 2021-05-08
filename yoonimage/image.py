import cv2
import numpy

from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D


class YoonImage:
    width: int
    height: int
    bpp: int
    __buffer: numpy.ndarray  # height, width, bpp(channel)

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, PLANE : {2}".format(self.width, self.height, self.bpp)

    def __init__(self,
                 pImage=None,
                 pBuffer: numpy.ndarray = None,
                 strFileName: str = None,
                 nWidth=640,
                 nHeight=480,
                 nBpp=1):
        if pImage is not None:
            assert isinstance(pImage, YoonImage)
            self.__buffer = pImage.copy_buffer()
            self.width = pImage.width
            self.height = pImage.height
            self.bpp = pImage.bpp
        elif pBuffer is not None:
            self.__buffer = pBuffer.copy()
            self.height, self.width, self.bpp = self.__buffer.shape
        elif strFileName is not None:
            self.__buffer = cv2.imread(strFileName)
            self.height, self.width, self.bpp = self.__buffer.shape
        else:
            self.width = nWidth
            self.height = nHeight
            self.bpp = nBpp
            self.__buffer = numpy.zeros((self.height, self.width, self.bpp), dtype=numpy.uint8)

    def get_buffer(self):
        return self.__buffer

    def get_tensor(self):
        return self.__buffer.transpose((2, 0, 1)).astype(numpy.float32)  # Channel, Y, X

    def __copy__(self):
        return YoonImage(pBuffer=self.__buffer)

    def copy_buffer(self):
        return self.__buffer.copy()

    def copy_tensor(self):
        return self.copy_buffer().transpose((2, 0, 1)).astype(numpy.float32)  # Channel, Y, X

    def normalization(self, dMean=0.5, dStd=0.5):
        return YoonImage(pBuffer=self.__buffer - dMean / dStd)

    def flip_horizontal(self):
        return YoonImage(pBuffer=numpy.flipud(self.__buffer))

    def flip_vertical(self):
        return YoonImage(pBuffer=numpy.fliplr(self.__buffer))

    def crop(self, pRect: YoonRect2D):
        assert isinstance(pRect, YoonRect2D)
        pResultBuffer = self.__buffer[int(pRect.top()): int(pRect.bottom()),
                        int(pRect.left()): int(pRect.right())].copy()
        return YoonImage(pBuffer=pResultBuffer)

    def resize(self, dScaleX: (int, float), dScaleY: (int, float)):
        pResultBuffer = cv2.resize(self.__buffer, None, fx=dScaleX, fy=dScaleY)
        return YoonImage(pBuffer=pResultBuffer)

    def draw_rectangle(self, pRect: YoonRect2D, pArrayColor: numpy.ndarray):
        cv2.rectangle(self.__buffer, (int(pRect.left()), int(pRect.top())), (int(pRect.right()), int(pRect.bottom())),
                      pArrayColor, 2)

    def draw_text(self, strText: str, pPos: YoonVector2D, pArrayColor: numpy.ndarray):
        cv2.putText(self.__buffer, strText, (int(pPos.x), int(pPos.y)), cv2.FONT_HERSHEY_PLAIN, 3, pArrayColor, 3)

    def show_image(self):
        cv2.imshow("Image", self.__buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
