from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D
from yoonpytory.line import YoonLine2D
from yoonimage.image import YoonImage


class YoonObject:
    label = 0
    score = 0.0
    pixel_count = 0
    object = None
    object_image = None

    def __init__(self, id: int, score: (int, float), pixel: int, object: (YoonRect2D, YoonLine2D, YoonVector2D),
                 image: YoonImage):
        self.label = id
        self.score = score
        self.pixel_count = pixel
        self.object = object.__copy__()
        self.object = image.__copy__()

    def __copy__(self):
        return YoonObject(id=self.label, score=self.score, pixel=self.pixel_count, object=self.object,
                          image=self.object_image)


def listing(objects: list, tag: str):
    result = []
    for object in objects:
        if isinstance(object, YoonObject):
            dic = {"id": object.label,
                   "score": object.score,
                   "pixel": object.pixel_count,
                   "object": object.object,
                   "image": object.object_image}
            result.append(dic[tag])
    return result
