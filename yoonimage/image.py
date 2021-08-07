import cv2
import cv2.cv2
import numpy

from yoonpytory.figure import YoonVector2D, YoonRect2D, YoonLine2D


class YoonImage:
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    width: int
    height: int
    channel: int
    __buffer: numpy.ndarray  # height, width, channel(bpp)
    """

    def __str__(self):
        return "WIDTH : {0}, HEIGHT : {1}, PLANE : {2}".format(self.width, self.height, self.channel)

    @classmethod
    def from_tensor(cls, pTensor: numpy.ndarray):
        # Change the transform and de-normalization
        return YoonImage(pBuffer=pTensor.transpose(1, 2, 0))

    @classmethod
    def from_array(cls,
                   pArray: numpy.ndarray,
                   nWidth: int,
                   nHeight: int,
                   nChannel: int,
                   strMode: "mix"  # mix, parallel
                   ):
        if strMode == "parallel":
            pArray = pArray.reshape(-1)
            pListBuffer = []
            for i in range(nChannel):
                pChannel = pArray[i * nWidth * nHeight: (i + 1) * nWidth * nHeight]
                pChannel = pChannel.reshape((nHeight, nWidth))
                pListBuffer.append(numpy.expand_dims(pChannel, axis=-1))
            return YoonImage(pBuffer=numpy.concatenate(pListBuffer, axis=-1))
        if strMode == "mix":
            return YoonImage(pBuffer=pArray.reshape((nHeight, nWidth, nChannel)))

    def __init__(self,
                 pImage=None,
                 pBuffer: numpy.ndarray = None,
                 strFileName: str = None,
                 nWidth=640,
                 nHeight=480,
                 nChannel=1):
        self.width: int
        self.height: int
        self.channel: int
        self.__buffer: numpy.ndarray  # height, width, channel(bpp)
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
            self.__buffer = cv2.cv2.imread(strFileName)  # load to BGR
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
        pResultBuffer = self.copy_buffer().astype(numpy.float32)
        pResultBuffer = (pResultBuffer - dMean) / dStd
        return dMean, dStd, YoonImage(pBuffer=pResultBuffer)

    def normalize(self, nChannel=None, dMean=128, dStd=255):
        if nChannel is None:
            return self._normalize_all(dMean, dStd)
        pResultBuffer = self.copy_buffer()
        pResultBuffer[:, :, nChannel] = (pResultBuffer[:, :, nChannel] - dMean) / dStd
        return YoonImage(pBuffer=pResultBuffer)

    def denormalize(self, nChannel=None, dMean=128, dStd=255):
        if nChannel is None:
            return self._denormalize_all(dMean, dStd)
        pResultBuffer = self.copy_buffer()
        pResultBuffer[:, :, nChannel] = pResultBuffer[:, :, nChannel] * dStd + dMean
        return YoonImage(pBuffer=pResultBuffer)

    def _normalize_all(self, dMean=128, dStd=255):
        pResultBuffer = self.copy_buffer()
        pResultBuffer = (pResultBuffer - dMean) / dStd
        return YoonImage(pBuffer=pResultBuffer)

    def _denormalize_all(self, dMean=128, dStd=255):
        pResultBuffer = self.copy_buffer()
        pResultBuffer = pResultBuffer * dStd + dMean
        return YoonImage(pBuffer=pResultBuffer)

    def pixel_decimal(self):
        return self._normalize_all(dMean=0, dStd=255)

    def pixel_recover(self):
        return self._denormalize_all(dMean=0, dStd=255)

    def to_binary(self, nThreshold=128):
        pResultBuffer = cv2.cv2.threshold(self.__buffer, nThreshold, 255, cv2.cv2.THRESH_BINARY)[1]
        return YoonImage(pBuffer=pResultBuffer)

    def to_gray(self):
        pResultBuffer = cv2.cv2.cvtColor(self.__buffer, cv2.cv2.COLOR_BGR2GRAY)
        return YoonImage(pBuffer=pResultBuffer)

    def to_color(self):
        pResultBuffer = cv2.cv2.cvtColor(self.__buffer, cv2.cv2.COLOR_GRAY2BGR)
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
        pResultBuffer = cv2.cv2.resize(self.__buffer, None, fx=dScaleX, fy=dScaleY)
        return YoonImage(pBuffer=pResultBuffer)

    def resize(self, nWidth: int, nHeight: int):
        if nWidth == self.width and nHeight == self.height:
            return self.__copy__()
        pResultBuffer = cv2.cv2.resize(self.__buffer, dsize=(nWidth, nHeight), interpolation=cv2.cv2.INTER_CUBIC)
        return YoonImage(pBuffer=pResultBuffer)

    # Resize image with unchanged aspect ratio using padding
    def resize_padding(self, nWidth: int, nHeight: int, nPadding: int = 128):
        if nWidth == self.width and nHeight == self.height:
            return self.__copy__()
        nWidthResized = int(self.width * min(nWidth / self.width, nHeight / self.height))
        nHeightResized = int(self.height * min(nWidth / self.width, nHeight / self.height))
        pBufferResized = cv2.cv2.resize(self.__buffer, dsize=(nWidthResized, nHeightResized),
                                        interpolation=cv2.cv2.INTER_CUBIC)
        nTop = (self.height - nHeightResized) // 2
        nLeft = (self.width - nWidthResized) // 2
        pCanvas = numpy.full((nHeight, nWidth, self.channel), nPadding)
        pCanvas[nTop:nTop + nHeightResized, nLeft:nLeft + nWidthResized, :] = pBufferResized
        return YoonImage(pBuffer=pCanvas)

    def rechannel(self, nChannel: int):
        if nChannel == self.channel:
            return self.__copy__()
        if nChannel == 1:
            return self.to_gray()
        elif nChannel == 3:
            return self.to_color()

    def draw_line(self,
                  pLine: YoonLine2D,
                  pArrayColor: numpy.ndarray,
                  nPenWidth: int = 3):
        if self.channel == 1:
            self.to_color()
        if pLine is not None:
            cv2.cv2.line(self.__buffer,
                         pt1=(pLine.startPos.to_tuple_int()),
                         pt2=(pLine.endPos.to_tuple_int()),
                         color=pArrayColor,
                         thickness=nPenWidth)

    def draw_rectangle(self,
                       pRect: YoonRect2D,
                       pArrayColor: numpy.ndarray,
                       nPenWidth: int = 3):
        if self.channel == 1:
            self.to_color()
        if pRect is not None:
            cv2.cv2.rectangle(self.__buffer,
                              pt1=(pRect.top_left().to_tuple_int()),
                              pt2=(pRect.bottom_right().to_tuple_int()),
                              color=pArrayColor,
                              thickness=nPenWidth)

    def draw_text(self, strText: str,
                  pPos: YoonVector2D,
                  pArrayColor: numpy.ndarray,
                  nScale: int = 3):
        if self.channel == 1:
            self.to_color()
        if pPos is not None:
            cv2.cv2.putText(self.__buffer,
                            text=strText,
                            org=(pPos.to_tuple_int()),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=nScale,
                            color=pArrayColor,
                            thickness=3)

    def show_image(self):
        cv2.imshow("Image", self.__buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
