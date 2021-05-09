from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D
from yoonpytory.line import YoonLine2D
from yoonimage.image import YoonImage


class YoonObject:
    label = 0
    name = ""
    score = 0.0
    pixelCount = 0
    region: (YoonRect2D, YoonLine2D, YoonVector2D) = None
    image: YoonImage = None

    def __init__(self,
                 nID: int = 0,
                 strName: str = "",
                 dScore=0.0,
                 nCountPixel: int = 0,
                 pRegion: (YoonVector2D, YoonLine2D, YoonRect2D) = None,
                 pImage: YoonImage = None):
        self.label = nID
        self.name = strName
        self.score = dScore
        self.pixelCount = nCountPixel
        if pRegion is not None:
            self.region = pRegion.__copy__()
        if pImage is not None:
            self.image = pImage.__copy__()

    def __copy__(self):
        return YoonObject(nID=self.label, strName=self.name, dScore=self.score, nCountPixel=self.pixelCount,
                          pRegion=self.region, pImage=self.image)


class YoonDataset:
    labels: list = []
    names: list = []
    scores: list = []
    pixelCounts: list = []
    regions: list = []
    buffers: list = []

    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self.labels)

    def __init__(self,
                 pList: list = None,
                 *args: (YoonObject, YoonImage, YoonRect2D, YoonLine2D, YoonVector2D)):
        if len(args) > 0:
            iCount = 0
            for pItem in args:
                if isinstance(pItem, YoonObject):
                    self.labels.append(pItem.label)
                    self.names.append(pItem.name)
                    self.scores.append(pItem.score)
                    self.pixelCounts.append(pItem.pixelCount)
                    self.regions.append(pItem.region.__copy__())
                    self.buffers.append(pItem.image.get_buffer().copy())
                elif isinstance(pItem, YoonImage):
                    self.labels.append(iCount)
                    self.buffers.append(pItem.get_buffer().copy())
                else:
                    self.labels.append(iCount)
                    self.regions.append(pItem.__copy__())
                iCount += 1
        else:
            if pList is not None:
                iCount = 0
                for pItem in pList:
                    if isinstance(pItem, YoonObject):
                        self.labels.append(pItem.label)
                        self.names.append(pItem.name)
                        self.scores.append(pItem.score)
                        self.pixelCounts.append(pItem.pixelCount)
                        self.regions.append(pItem.region.__copy__())
                        self.buffers.append(pItem.image.get_buffer().copy())
                    elif isinstance(pItem, YoonImage):
                        self.labels.append(iCount)
                        self.buffers.append(pItem.get_buffer().copy())
                    elif isinstance(pItem, (YoonRect2D, YoonLine2D, YoonVector2D)):
                        self.labels.append(iCount)
                        self.regions.append(pItem.__copy__())
                    iCount += 1

    def __copy__(self):
        pResult = YoonDataset()
        pResult.labels = self.labels.copy()
        pResult.names = self.names.copy()
        pResult.scores = self.scores.copy()
        pResult.pixelCounts = self.pixelCounts.copy()
        pResult.regions = self.regions.copy()
        pResult.buffers = self.buffers.copy()
        return pResult

    def __getitem__(self, item: int):
        nLabel: int = 0
        strName: str = ""
        dScore: float = 0.00
        nPixelCount: int = 0
        pRegion = None
        pImage: YoonImage = None
        if 0 <= item < len(self.labels):
            nLabel = self.labels[item]
        if 0 <= item < len(self.names):
            strName = self.names[item]
        if 0 <= item < len(self.scores):
            dScore = self.scores[item]
        if 0 <= item < len(self.pixelCounts):
            nPixelCount = self.pixelCounts[item]
        if 0 <= item < len(self.regions):
            pRegion = self.regions[item]
        if 0 <= item < len(self.buffers):
            pImage = YoonImage(pBuffer=self.buffers[item])
        return YoonObject(nID=nLabel, strName=strName, dScore=dScore, nCountPixel=nPixelCount,
                          pRegion=pRegion, pImage=pImage)

    def __setitem__(self, key: int, value: YoonObject):
        if 0 <= key < len(self.labels):
            self.labels[key] = value.label
        if 0 <= key < len(self.names):
            self.names[key] = value.name
        if 0 <= key < len(self.scores):
            self.scores[key] = value.score
        if 0 <= key < len(self.pixelCounts):
            self.pixelCounts[key] = value.pixelCount
        if 0 <= key < len(self.regions):
            self.regions[key] = value.region
        if 0 <= key < len(self.buffers):
            self.buffers[key] = value.image.get_buffer()

    def clear(self):
        self.labels.clear()
        self.names.clear()
        self.scores.clear()
        self.pixelCounts.clear()
        self.regions.clear()
        self.buffers.clear()

    def append(self, pObject: YoonObject):
        self.labels.append(pObject.label)
        self.names.append(pObject.name)
        self.scores.append(pObject.score)
        self.pixelCounts.append(pObject.pixelCount)
        self.regions.append(pObject.region.__copy__())
        self.buffers.append(pObject.image.get_buffer().copy())

    def to_region_points(self):
        pListResult = []
        for pRegion in self.regions:
            pListResult.append(pRegion.to_list())
        return pListResult