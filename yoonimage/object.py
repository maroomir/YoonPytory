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

    def __init__(self, **kwargs):
        if kwargs.get("id"):
            self.label = kwargs["id"]
        if kwargs.get("score"):
            self.score = kwargs["score"]
        if kwargs.get("pixel"):
            self.pixel_count = kwargs["pixel"]
        if kwargs.get("object"):
            assert isinstance(kwargs["object"], (YoonVector2D, YoonRect2D, YoonLine2D))
            self.object = kwargs["object"].__copy__()
        if kwargs.get("image"):
            assert isinstance(kwargs["image"], YoonImage)
            self.object_image = kwargs["image"].__copy__()

    def __copy__(self):
        return YoonObject(id=self.label, score=self.score, pixel=self.pixel_count, object=self.object,
                          image=self.object_image)

    @staticmethod
    def listing(listObject: list, strTag: str):
        listResult = []
        for content in listObject:
            if isinstance(content, YoonObject):
                dic = {"id": content.label,
                       "score": content.score,
                       "pixel": content.pixel_count,
                       "object": content.object,
                       "object_to_list": content.object.to_list(),
                       "object_to_tuple": content.object.to_tuple(),
                       "image": content.object_image}
                listResult.append(dic[strTag])
        return listResult
