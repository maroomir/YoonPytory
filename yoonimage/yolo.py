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


def detection(pImageSource: YoonImage, pNet: YoloNet, pSize: tuple, dScale: float, dScoreTarget=0.5):
    # Initialize input blobs for leaning network
    pArrayBlob = cv2.dnn.blobFromImage(pImageSource.get_buffer(), scalefactor=dScale, size=pSize, mean=(0, 0, 0),
                                       swapRB=True, crop=False)
    # Set network input
    pNet.net.setInput(pArrayBlob)
    # Run network
    pListResult = pNet.net.forward(pNet.layers)
    # Analyze network result
    pListObject = []
    for pResult in pListResult:
        for arrayInfo in pResult:  # Information array : CenterX, CenterY, Width, Height, Scores ...
            scores = arrayInfo[5:]
            nIDMax = numpy.argmax(scores)
            dScoreMax = scores[nIDMax]
            if dScoreMax > dScoreTarget:
                pRectMax = YoonRect2D(dX=arrayInfo[0] * pImageSource.width, dY=arrayInfo[1] * pImageSource.height,
                                      dWidth=arrayInfo[2] * pImageSource.width,
                                      dHeight=arrayInfo[3] * pImageSource.height)
                # score type is float64 because of fixing error in remove_noise
                pListObject.append(YoonObject(nID=nIDMax, dScore=numpy.float64(dScoreMax), pObject=pRectMax,
                                              pImage=pImageSource.crop(pRectMax)))
    return pListObject


def remove_noise(pListObject: list):
    pListRect = YoonObject.listing(pListObject, "object_to_list")
    pListScore = YoonObject.listing(pListObject, "score")
    pArrayResult = cv2.dnn.NMSBoxes(pListRect, pListScore, score_threshold=0.5, nms_threshold=0.4)
    pListResult = []
    for i in range(len(pListObject)):
        if i in pArrayResult:
            pListResult.append(pListObject[i])
    return pListResult


def draw_detection_result(pListObject: list, pImage: YoonImage, pNet: YoloNet):
    for pObject in pListObject:
        if isinstance(pObject, YoonObject):
            nID = pObject.label
            pImage.draw_rectangle(pObject.object, pArrayColor=pNet.colors[nID])
            pImage.draw_text(pNet.classes[nID], pObject.object.top_left(), pArrayColor=pNet.colors[nID])
    pImage.show_image()
