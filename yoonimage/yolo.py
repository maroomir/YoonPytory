import cv2
import numpy
from yoonimage.image import YoonImage
from yoonimage.object import YoonObject
from yoonpytory.rect import YoonRect2D
from yoonimage.object import listing


class YoloParameter():
    net = None
    classes = []
    layers = []
    colors = []

    def is_enable(self):
        return self.net is not None

    def load_modern_net(self, weight_file: str, config_file: str, names_file: str):
        self.net = cv2.dnn.readNet(weight_file, config_file)
        with open(names_file, "r") as file:
            for line in file.readlines():
                self.classes.append(line.strip())
        list_name = self.net.getLayerNames()
        for layer in self.net.getUnconnectedOutLayers():
            self.layers.append(list_name[layer[0] - 1])
        self.colors = numpy.random.uniform(0, 255, size=(len(self.classes), 3))


def detection(image: YoonImage, param: YoloParameter, thres_score: float):
    blobs = cv2.dnn.blobFromImage(image.get_buffer(), scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    param.net.setInput(blobs)
    results = param.net.forward(param.layers)
    object_list = []
    for feature in results:
        for info_array in feature:    # Information Array : CenterX, CenterY, Width, Height, Socres...
            scores = info_array[5:]
            max_id = numpy.argmax(scores)
            max_score = scores[max_id]
            if max_score > thres_score:
                rect = YoonRect2D(x=info_array[0]*image.width, y=info_array[1]*image.height, width=info_array[2]*image.width, height=info_array[3]*image.height)
                object_list.append(YoonObject(id=max_id, score=max_score, object=rect, image=image.crop(rect)))
    cv2.dnn.NMSBoxes(listing(object_list, "object"), listing(object_list, "score"), score_threshold=thres_score, nms_threshold=0.4)
