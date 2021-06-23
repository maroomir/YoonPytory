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

    @classmethod
    def from_tensor(cls, pTensor: numpy.ndarray):
        # Change the transform and de-normalization
        return YoonImage(pBuffer=pTensor.transpose(1, 2, 0))

    @classmethod
    def from_mats(cls, *args):
        pListMat = []
        if len(args) > 0:
            for i in range(len(args)):
                assert isinstance(args[i], numpy.ndarray)
                pListMat.append(args)
            return numpy.array(pListMat)

    @classmethod
    def from_array(cls,
                   pArray: numpy.ndarray,
                   nWidth: int,
                   nHeight: int,
                   nChannel: int):
        return YoonImage(pBuffer=pArray.reshape((nHeight, nWidth, nChannel)))

    def __init__(self,
                 pImage=None,
                 pBuffer: numpy.ndarray = None,
                 strFileName: str = None,
                 nWidth=640,
                 nHeight=480,
                 nChannel=1):
        if pImage is not None:
            assert isinstance(pImage, YoonImage)
            self.__buffer = pImage.copy_buffer()
            if len(self.__buffer.shape) < 3:  # Contains only the height and width
                self.__buffer = numpy.expand_dims(self.__buffer, axis=-1)
            self.height, self.width, self.channel = self.__buffer.shape
        elif pBuffer is not None:
            self.__buffer = pBuffer.copy()
            if len(self.__buffer.shape) < 3:  # Contains only the height and width
                self.__buffer = numpy.expand_dims(self.__buffer, axis=-1)
            self.height, self.width, self.channel = self.__buffer.shape
        elif strFileName is not None:
            self.__buffer = cv2.imread(strFileName)  # load to BGR
            if len(self.__buffer.shape) < 3:  # Contains only the height and width
                self.__buffer = numpy.expand_dims(self.__buffer, axis=-1)
            self.height, self.width, self.channel = self.__buffer.shape
        else:
            self.width = nWidth
            self.height = nHeight
            self.channel = nChannel
            self.__buffer = numpy.zeros((self.height, self.width, self.channel), dtype=numpy.uint8)

    def get_buffer(self):
        return self.__buffer

    def get_tensor(self):
        # Change the transform to (Channel, Height, Width)
        return self.__buffer.transpose((2, 0, 1)).astype(numpy.float32)

    def __copy__(self):
        return YoonImage(pBuffer=self.__buffer)

    def copy_buffer(self):
        return self.__buffer.copy()

    def copy_tensor(self):
        # Change the transform to (Channel, Height, Width)
        return self.copy_buffer().transpose((2, 0, 1)).astype(numpy.float32)

    def minmax_normalize(self):
        nMax = numpy.max(self.__buffer)
        nMin = numpy.min(self.__buffer)
        pReseultBuffer = self.copy_buffer().astype(numpy.float32)
        pReseultBuffer = (pReseultBuffer - nMin) / (nMax - nMin)
        return nMin, (nMax - nMin), YoonImage(pBuffer=pReseultBuffer)

    def z_normalize(self):
        pFuncMask = self.__buffer > 0
        dMean = numpy.mean(self.__buffer[pFuncMask])
        dStd = numpy.std(self.__buffer[pFuncMask])
        pReseultBuffer = self.copy_buffer().astype(numpy.float32)
        pReseultBuffer = numpy.matmul((pReseultBuffer - dMean) / dStd, pFuncMask)
        return dMean, dStd, YoonImage(pBuffer=pReseultBuffer)

    def normalize(self, nChannel=None, dMean=128, dStd=255):
        if nChannel is None:
            return self._normalize_all(dMean, dStd)
        return YoonImage(pBuffer=self.__buffer[:, :, nChannel] - dMean / dStd)

    def denormalize(self, nChannel=None, dMean=128, dStd=255):
        if nChannel is None:
            return self._denormalize_all(dMean, dStd)
        return YoonImage(pBuffer=self.__buffer[:, :, nChannel] * dStd + dMean)

    def _normalize_all(self, dMean=128, dStd=255):
        return YoonImage(pBuffer=self.__buffer - dMean / dStd)

    def _denormalize_all(self, dMean=128, dStd=255):
        return YoonImage(pBuffer=self.__buffer * dStd + dMean)

    def pixel_decimal(self):
        return self._normalize_all(dMean=0, dStd=255)

    def pixel_recover(self):
        return self._denormalize_all(dMean=0, dStd=255)

    def to_binary(self, nThreshold=128):
        pResultBuffer = cv2.threshold(self.__buffer, nThreshold, 255, cv2.THRESH_BINARY)[1]
        return YoonImage(pBuffer=pResultBuffer)

    def to_gray(self):
        pResultBuffer = cv2.cvtColor(self.__buffer, cv2.COLOR_BGR2GRAY)
        return YoonImage(pBuffer=pResultBuffer)

    def to_color(self):
        pResultBuffer = cv2.cvtColor(self.__buffer, cv2.COLOR_GRAY2BGR)
        return YoonImage(pBuffer=pResultBuffer)

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

    def rechannel(self, nChannel: int):
        if nChannel == self.channel:
            return self.__copy__()
        if nChannel == 1:
            return self.to_gray()
        elif nChannel == 3:
            return self.to_color()

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
