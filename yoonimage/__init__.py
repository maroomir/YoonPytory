from yoonimage.image import YoonImage as yimage
from yoonimage.object import YoonObject as yobject
from yoonimage.yolo import YoloNet as yolonet
from yoonimage.yolo import *

YOLO_SCALE_ONE_ZERO_PER_8BIT = 1/255
YOLO_SIZE_FAST = (320, 320)
YOLO_SIZE_NORMAL = (416, 416)
YOLO_SIZE_MAX = (609, 609)