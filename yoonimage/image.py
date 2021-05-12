import cv2
import numpy

from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D


class YoonImage:
    width: int
    height: int
    channel: int
    __buffer: numpy.ndarray  # height, width, channel(bpp)

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, PLANE : {2}".format(self.width, self.height, self.channel)

    @staticmethod
    def from_tensor(pTensor: numpy.ndarray, bNormalized: bool = True):
        # Change the transform and de-normalization
        return YoonImage(pBuffer=pTensor.transpose((1, 2, 0)) * (255 if bNormalized is True else 1).astype(numpy.int))

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
            self.channel = pImage.channel
        elif pBuffer is not None:
            self.__buffer = pBuffer.copy()
            self.height, self.width, self.channel = self.__buffer.shape
        elif strFileName is not None:
            self.__buffer = cv2.imread(strFileName)  # load to BGR
            self.height, self.width, self.channel = self.__buffer.shape
        else:
            self.width = nWidth
            self.height = nHeight
            self.channel = nBpp
            self.__buffer = numpy.zeros((self.height, self.width, self.channel), dtype=numpy.uint8)

    def get_buffer(self):
        return self.__buffer

    def get_tensor(self, bNormalize: bool = True):
        # Change the transform to (Channel, Height, Width) and normalization
        return self.__buffer.transpose((2, 0, 1)).astype(numpy.float32) / (255.0 if bNormalize is True else 1.0)

    def __copy__(self):
        return YoonImage(pBuffer=self.__buffer)

    def copy_buffer(self):
        return self.__buffer.copy()

    def copy_tensor(self, bNormalize: bool = True):
        # Change the transform to (Channel, Height, Width) and normalization
        return self.copy_buffer().transpose((2, 0, 1)).astype(numpy.float32) / (255.0 if bNormalize is True else 1.0)

    def normalization(self, dMean=0, dStd=1):
        return YoonImage(pBuffer=self.__buffer - dMean / dStd)

    def de_normalization(self, dMean=0, dStd=1):
        return YoonImage(pBuffer=self.__buffer * dStd + dMean)

    def binary(self, nThreshold=128):
        return YoonImage(pBuffer=cv2.threshold(self.__buffer, nThreshold, 255, cv2.THRESH_BINARY))

    def flip_horizontal(self):
        return YoonImage(pBuffer=numpy.flipud(self.__buffer))

    def flip_vertical(self):
        return YoonImage(pBuffer=numpy.fliplr(self.__buffer))

    def crop(self, pRect: YoonRect2D):
        assert isinstance(pRect, YoonRect2D)
        pResultBuffer = self.__buffer[int(pRect.top()): int(pRect.bottom()),
                        int(pRect.left()): int(pRect.right())].copy()
        return YoonImage(pBuffer=pResultBuffer)

    def scale(self, dScaleX: (int, float), dScaleY: (int, float)):
        pResultBuffer = cv2.resize(self.__buffer, None, fx=dScaleX, fy=dScaleY)
        return YoonImage(pBuffer=pResultBuffer)

    def resize(self, nWidth: int, nHeight: int):
        if nWidth == self.width and nHeight == self.height:
            return self.__copy__()
        pResultBuffer = cv2.resize(self.__buffer, dsize=(nWidth, nHeight), interpolation=cv2.INTER_CUBIC)
        return YoonImage(pBuffer=pResultBuffer)

    # Resize image with unchanged aspect ratio using padding
    def resize_with_padding(self, nWidth: int, nHeight: int, nPadding: int = 128):
        nWidthResized = int(self.width * min(nWidth / self.width, nHeight / self.height))
        nHeightResized = int(self.height * min(nWidth / self.width, nHeight / self.height))
        pBufferResized = cv2.resize(self.__buffer, dsize=(nWidthResized, nHeightResized), interpolation=cv2.INTER_CUBIC)
        nTop = (self.height - nHeightResized) // 2
        nLeft = (self.width - nWidthResized) // 2
        pCanvas = numpy.full((nHeight, nWidth, self.channel), nPadding)
        pCanvas[nTop:nTop + nHeightResized, nLeft:nLeft + nWidthResized, :] = pBufferResized
        return YoonImage(pBuffer=pCanvas)

    def draw_rectangle(self, pRect: YoonRect2D, pArrayColor: numpy.ndarray):
        cv2.rectangle(self.__buffer, (int(pRect.left()), int(pRect.top())), (int(pRect.right()), int(pRect.bottom())),
                      pArrayColor, 2)

    def draw_text(self, strText: str, pPos: YoonVector2D, pArrayColor: numpy.ndarray):
        cv2.putText(self.__buffer, strText, (int(pPos.x), int(pPos.y)), cv2.FONT_HERSHEY_PLAIN, 3, pArrayColor, 3)

    def show_image(self):
        cv2.imshow("Image", self.__buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
