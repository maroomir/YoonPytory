from yoonimage.image import YoonImage as image
from yoonimage.object import YoonObject as object
from yoonimage.objectDetection.yolo import YoloNet
from yoonimage.objectDetection.yolo import *

YOLO_SCALE_ONE_ZERO_PER_8BIT = 1/255
YOLO_SIZE_FAST = (320, 320)
YOLO_SIZE_NORMAL = (416, 416)
YOLO_SIZE_MAX = (609, 609)