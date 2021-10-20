import cv2
import cv2.cv2
import numpy
from numpy import ndarray

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
    def from_buffer(cls, buffer: numpy.ndarray):
        image = YoonImage()
        image.__buffer = buffer.copy()
        if len(image.__buffer.shape) < 3:  # Contains only the height and width
            image.__buffer = numpy.expand_dims(image.__buffer, axis=-1)
        image.height, image.width, image.channel = image.__buffer.shape
        return image

    @classmethod
    def from_tensor(cls,
                    tensor: numpy.ndarray):
        # Change the transform and de-normalization
        return YoonImage.from_buffer(tensor.transpose(1, 2, 0))

    @classmethod
    def from_image(cls, image):
        assert isinstance(image, YoonImage)
        return YoonImage.from_buffer(image.__buffer)

    @classmethod
    def from_path(cls, file_path: str):
        buffer = cv2.cv2.imread(file_path)  # load to BGR
        return YoonImage.from_buffer(buffer)

    @classmethod
    def parse_array(cls,
                    width: int,
                    height: int,
                    channel: int,
                    array: ndarray,
                    mode="mix"  # mix, parallel
                    ):
        if mode == "parallel":
            array = array.reshape(-1)
            buffers = []
            for ch in range(channel):
                ch_buffer = array[ch * width * height: (ch + 1) * width * height]
                ch_buffer = ch_buffer.reshape((height, width))
                buffers.append(numpy.expand_dims(ch_buffer, axis=-1))
            result = numpy.concatenate(buffers, axis=-1)
            return YoonImage.from_buffer(result)
        elif mode == "mix":
            return YoonImage.from_buffer(array.reshape((height, width, channel)))

    def __init__(self, width=640, height=480, channel=1):
        self.width = width
        self.height = height
        self.channel = channel
        self.__buffer = numpy.zeros((self.height, self.width, self.channel), dtype=numpy.uint8)

    def get_buffer(self):
        return self.__buffer

    def get_tensor(self):
        # Change the transform to (Channel, Height, Width)
        return self.__buffer.transpose((2, 0, 1)).astype(numpy.float32)

    def __copy__(self):
        return YoonImage.from_buffer(self.__buffer)

    def copy_buffer(self):
        return self.__buffer.copy()

    def copy_tensor(self):
        # Change the transform to (Channel, Height, Width)
        return self.copy_buffer().transpose((2, 0, 1)).astype(numpy.float32)

    def minmax_normalize(self):
        max_value = numpy.max(self.__buffer)
        min_value = numpy.min(self.__buffer)
        result = self.__buffer.astype(numpy.float32)
        result = (result - min_value) / (max_value - min_value)
        return min_value, (max_value - min_value), YoonImage.from_buffer(result)

    def z_normalize(self):
        mask_func = self.__buffer > 0
        mean = numpy.mean(self.__buffer[mask_func])
        std = numpy.std(self.__buffer[mask_func])
        result = self.__buffer.astype(numpy.float32)
        result = (result - mean) / std
        return mean, std, YoonImage.from_buffer(result)

    def normalize(self, channel=None, mean=128, std=255):
        if channel is None:
            return self._normalize_all(mean, std)
        result = self.__buffer
        result[:, :, channel] = (result[:, :, channel] - mean) / std
        return YoonImage.from_buffer(result)

    def denormalize(self, channel=None, mean=128, std=255):
        if channel is None:
            return self._denormalize_all(mean, std)
        result = self.__buffer
        result[:, :, channel] = result[:, :, channel] * std + mean
        return YoonImage.from_buffer(result)

    def _normalize_all(self, mean=128, std=255):
        result = self.__buffer
        result = (result - mean) / std
        return YoonImage.from_buffer(result)

    def _denormalize_all(self, mean=128, std=255):
        result = self.__buffer
        result = result * std + mean
        return YoonImage.from_buffer(result)

    def pixel_decimal(self):
        return self._normalize_all(mean=0, std=255)

    def pixel_recover(self):
        return self._denormalize_all(mean=0, std=255)

    def to_binary(self, thresh=128):
        result = cv2.cv2.threshold(self.__buffer, thresh, 255, cv2.cv2.THRESH_BINARY)[1]
        return YoonImage.from_buffer(result)

    def to_gray(self):
        result = cv2.cv2.cvtColor(self.__buffer, cv2.cv2.COLOR_BGR2GRAY)
        return YoonImage.from_buffer(result)

    def to_color(self):
        result = cv2.cv2.cvtColor(self.__buffer, cv2.cv2.COLOR_GRAY2BGR)
        return YoonImage.from_buffer(result)

    def flip_horizontal(self):
        return YoonImage.from_buffer(numpy.flipud(self.__buffer))

    def flip_vertical(self):
        return YoonImage.from_buffer(numpy.fliplr(self.__buffer))

    def crop(self, rect: YoonRect2D):
        assert isinstance(rect, YoonRect2D)
        result = self.__buffer[int(rect.top()): int(rect.bottom()), int(rect.left()): int(rect.right())]
        return YoonImage.from_buffer(result)

    def scale(self, scale_x: (int, float), scale_y: (int, float)):
        result = cv2.cv2.resize(self.__buffer, None, fx=scale_x, fy=scale_y)
        return YoonImage.from_buffer(result)

    def resize(self, width: int, height: int):
        if width == self.width and height == self.height:
            return self.__copy__()
        result = cv2.cv2.resize(self.__buffer, dsize=(width, height), interpolation=cv2.cv2.INTER_CUBIC)
        return YoonImage.from_buffer(result)

    # Resize image with unchanged aspect ratio using padding
    def resize_padding(self, width: int, height: int, pad: int = 128):
        if width == self.width and height == self.height:
            return self.__copy__()
        width_resized = int(self.width * min(width / self.width, height / self.height))
        height_resized = int(self.height * min(width / self.width, height / self.height))
        result = cv2.cv2.resize(self.__buffer, dsize=(width_resized, height_resized), interpolation=cv2.cv2.INTER_CUBIC)
        top = (self.height - height_resized) // 2
        left = (self.width - width_resized) // 2
        canvas = numpy.full((height, width, self.channel), pad)
        canvas[top:top + height_resized, left:left + width_resized, :] = result
        return YoonImage.from_buffer(canvas)

    def rechannel(self, channel: int):
        if channel == self.channel:
            return self.__copy__()
        if channel == 1:
            return self.to_gray()
        elif channel == 3:
            return self.to_color()

    def draw_line(self,
                  line: YoonLine2D,
                  colors: numpy.ndarray,
                  pen_w: int = 3):
        if self.channel == 1:
            self.to_color()
        if line is not None:
            cv2.cv2.line(self.__buffer,
                         pt1=(line.start_pos.to_tuple_int()),
                         pt2=(line.end_pos.to_tuple_int()),
                         color=colors,
                         thickness=pen_w)

    def draw_rectangle(self,
                       rect: YoonRect2D,
                       colors: numpy.ndarray,
                       pen_w: int = 3):
        if self.channel == 1:
            self.to_color()
        if rect is not None:
            cv2.cv2.rectangle(self.__buffer,
                              pt1=(rect.top_left().to_tuple_int()),
                              pt2=(rect.bottom_right().to_tuple_int()),
                              color=colors,
                              thickness=pen_w)

    def draw_text(self,
                  text: str,
                  pos: YoonVector2D,
                  colors: numpy.ndarray,
                  scale: int = 3):
        if self.channel == 1:
            self.to_color()
        if pos is not None:
            cv2.cv2.putText(self.__buffer,
                            text=text,
                            org=(pos.to_tuple_int()),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=scale,
                            color=colors,
                            thickness=3)

    def show_image(self):
        cv2.imshow("Image", self.__buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
