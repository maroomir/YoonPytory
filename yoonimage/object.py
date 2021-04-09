from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D
from yoonpytory.line import YoonLine2D
from yoonimage.image import YoonImage


class YoonObject:
    label = 0
    score = 0.0
    pixelCount = 0
    object = None
    objectImage = None

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
            self.object = pObject.__copy__()
        if pImage is not None:
            self.objectImage = pImage.__copy__()

    def __copy__(self):
        return YoonObject(id=self.label, score=self.score, pixel=self.pixelCount, object=self.object,
                          image=self.objectImage)

    @staticmethod
    def listing(pListObject: list, strTag: str):
        pListResult = []
        for content in pListObject:
            if isinstance(content, YoonObject):
                dic = {"id": content.label,
                       "score": content.score,
                       "pixel": content.pixelCount,
                       "object": content.object,
                       "object_to_list": content.object.to_list(),
                       "object_to_tuple": content.object.to_tuple(),
                       "image": content.objectImage}
                pListResult.append(dic[strTag])
        return pListResult
