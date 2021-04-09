import cv2
import numpy
from yoonimage.image import YoonImage
from yoonimage.object import YoonObject
from yoonpytory.rect import YoonRect2D


class YoloNet:
    net = None
    classes = []
    layers = []
    colors = []

    def is_enable(self):
        return self.net is not None

    def load_modern_net(self, strWeightFile: str, strConfigFile: str, strNamesFile: str):
        self.net = cv2.dnn.readNet(strWeightFile, strConfigFile)
        with open(strNamesFile, "r") as file:
            for line in file.readlines():
                self.classes.append(line.strip())
        listName = self.net.getLayerNames()
        for layer in self.net.getUnconnectedOutLayers():
            self.layers.append(listName[layer[0] - 1])
        self.colors = numpy.random.uniform(0, 255, size=(len(self.classes), 3))


def detection(pImage: YoonImage, pNet: YoloNet, pSize: tuple, dScale: float, dScoreTarget=0.5):
    # Initialize input blobs for leaning network
    arrayBlob = cv2.dnn.blobFromImage(pImage.get_buffer(), scalefactor=dScale, size=pSize, mean=(0, 0, 0),
                                      swapRB=True, crop=False)
    # Set network input
    pNet.net.setInput(arrayBlob)
    # Run network
    listResult = pNet.net.forward(pNet.layers)
    # Analyze network result
    listObject = []
    for pResult in listResult:
        for arrayInfo in pResult:  # Information array : CenterX, CenterY, Width, Height, Scores ...
            scores = arrayInfo[5:]
            max_id = numpy.argmax(scores)
            max_score = scores[max_id]
            if max_score > dScoreTarget:
                rect = YoonRect2D(x=arrayInfo[0] * pImage.width, y=arrayInfo[1] * pImage.height,
                                  width=arrayInfo[2] * pImage.width, height=arrayInfo[3] * pImage.height)
                listObject.append(YoonObject(id=max_id, score=numpy.float64(max_score), object=rect, image=pImage.crop(
                    rect)))  # score type is float64 because of fixing error in remove_noise
    return listObject


def remove_noise(listObject: list):
    listRect = YoonObject.listing(listObject, "object_to_list")
    listScore = YoonObject.listing(listObject, "score")
    results = cv2.dnn.NMSBoxes(listRect, listScore, score_threshold=0.5, nms_threshold=0.4)
    listResult = []
    for i in range(len(listObject)):
        if i in results:
            listResult.append(listObject[i])
    return listResult


def draw_detection_result(listObject: list, pImage: YoonImage, pNet: YoloNet):
    for pObject in listObject:
        if isinstance(pObject, YoonObject):
            object_id = pObject.label
            pImage.draw_rectangle(pObject.object, arrayColor=pNet.colors[object_id])
            pImage.draw_text(pNet.classes[object_id], pObject.object.top_left(), arrayColor=pNet.colors[object_id])
    pImage.show_image()
