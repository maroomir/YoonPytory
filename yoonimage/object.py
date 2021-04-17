from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D
from yoonpytory.line import YoonLine2D
from yoonimage.image import YoonImage


class YoonObject:
    label = 0
    score = 0.0
    pixelCount = 0
    area = None
    cropImage = None

    def __init__(self,
                 nID: int = 0,
                 dScore=0.0,
                 nCountPixel: int = 0,
                 pObject: (YoonVector2D, YoonLine2D, YoonRect2D) = None,
                 pImage: YoonImage = None):
        self.label = nID
        self.score = dScore
        self.pixelCount = nCountPixel
        if pObject is not None:
            self.area = pObject.__copy__()
        if pImage is not None:
            self.cropImage = pImage.__copy__()

    def __copy__(self):
        return YoonObject(id=self.label, score=self.score, pixel=self.pixelCount, object=self.area,
                          image=self.cropImage)

    @staticmethod
    def listing(pList: list, strTag: str):
        pListResult = []
        for pObject in pList:
            if isinstance(pObject, YoonObject):
                pDic = {"id": pObject.label,
                        "score": pObject.score,
                        "pixel": pObject.pixelCount,
                        "area": pObject.area,
                        "area_to_list": pObject.area.to_list(),
                        "area_to_tuple": pObject.area.to_tuple(),
                        "image": pObject.cropImage}
                pListResult.append(pDic[strTag])
        return pListResult
