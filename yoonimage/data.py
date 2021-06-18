import numpy
import matplotlib.pyplot

from yoonpytory.rect import YoonRect2D
from yoonpytory.vector import YoonVector2D
from yoonpytory.line import YoonLine2D
from yoonimage.image import YoonImage


class YoonObject:
    label = 0
    name = ""
    score = 0.0
    pix_count = 0
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
        self.pix_count = nCountPixel
        if pRegion is not None:
            self.region = pRegion.__copy__()
        if pImage is not None:
            self.image = pImage.__copy__()

    def __copy__(self):
        return YoonObject(nID=self.label, strName=self.name, dScore=self.score, nCountPixel=self.pix_count,
                          pRegion=self.region, pImage=self.image)


class YoonDataset:
    labels: list = []
    names: list = []
    scores: list = []
    pix_counts: list = []
    regions: list = []
    images: list = []

    @staticmethod
    def from_tensor(pTensor: numpy.ndarray,
                    bWithBatch: bool = True,
                    nChannel: int = 3):
        pDataSet = YoonDataset()
        if bWithBatch:
            for iBatch in range(len(pTensor)):
                pDataSet.labels.append(iBatch)
                pDataSet.images.append(YoonImage.from_tensor(pTensor=pTensor[iBatch]))
            return pDataSet
        else:
            for iBatch in range(len(pTensor), nChannel):
                pDataSet.labels.append(iBatch)
                pDataSet.images.append(YoonImage.from_tensor(pTensor=numpy.concatenate([pTensor[iBatch + i]
                                                                                        for i in range(nChannel)])))
            return pDataSet

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
                    self.pix_counts.append(pItem.pix_count)
                    self.regions.append(pItem.region.__copy__())
                    self.images.append(pItem.image.__copy__())
                elif isinstance(pItem, YoonImage):
                    self.labels.append(iCount)
                    self.images.append(pItem.__copy__())
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
                        self.pix_counts.append(pItem.pix_count)
                        self.regions.append(pItem.region.__copy__())
                        self.images.append(pItem.image.__copy__())
                    elif isinstance(pItem, YoonImage):
                        self.labels.append(iCount)
                        self.images.append(pItem.__copy__())
                    elif isinstance(pItem, (YoonRect2D, YoonLine2D, YoonVector2D)):
                        self.labels.append(iCount)
                        self.regions.append(pItem.__copy__())
                    iCount += 1

    def __copy__(self):
        pResult = YoonDataset()
        pResult.labels = self.labels.copy()
        pResult.names = self.names.copy()
        pResult.scores = self.scores.copy()
        pResult.pix_counts = self.pix_counts.copy()
        pResult.regions = self.regions.copy()
        pResult.images = self.images.copy()
        return pResult

    def __getitem__(self, item):
        def get_object(i,
                       nIndexStop: int = None,
                       pRemained=None,
                       bProcessing=False):
            if isinstance(i, tuple):
                pListIndex = list(i)[1:] + [None]
                pListResult = []
                for iIndex, iRemain in zip(i, pListIndex):
                    pResult, nIndexStop = get_object(iIndex, nIndexStop, iRemain, True)
                    pListResult.append(pResult)
                return pListResult
            elif isinstance(i, slice):
                pRange = range(*i.indices(len(self)))
                pListResult = [get_object(j) for j in pRange]
                if bProcessing:
                    return pListResult, pRange[-1]
                else:
                    return pListResult
            elif i is Ellipsis:
                if nIndexStop is not None:
                    nIndexStop += 1
                nIndexEnd = pRemained
                if isinstance(pRemained, slice):
                    nIndexEnd = pRemained.start
                pResult = get_object(slice(nIndexStop, nIndexEnd), bProcessing=True)
                if bProcessing:
                    return pResult[0], pResult[1]
                else:
                    return pResult[0]
            else:
                nLabel: int = 0
                strName: str = ""
                dScore: float = 0.00
                nPixelCount: int = 0
                pRegion = None
                pImage: YoonImage = None
                if i > len(self):
                    raise IndexError("Index is too big")
                elif i < 0:
                    raise IndexError("Index is abnormal")
                if 0 <= i < len(self.labels):
                    nLabel = self.labels[i]
                if 0 <= i < len(self.names):
                    strName = self.names[i]
                if 0 <= i < len(self.scores):
                    dScore = self.scores[i]
                if 0 <= i < len(self.pix_counts):
                    nPixelCount = self.pix_counts[i]
                if 0 <= i < len(self.regions):
                    pRegion = self.regions[i]
                if 0 <= i < len(self.images):
                    pImage = self.images[i]
                pObject = YoonObject(nID=nLabel, strName=strName, dScore=dScore, nCountPixel=nPixelCount,
                                     pRegion=pRegion, pImage=pImage)
                if bProcessing:
                    return pObject, i
                else:
                    return pObject

        pResultItem = get_object(item)
        if isinstance(pResultItem, list):
            return YoonDataset(pList=pResultItem)
        else:
            return pResultItem

    def __setitem__(self, key: int, value: YoonObject):
        if 0 <= key < len(self.labels):
            self.labels[key] = value.label
        if 0 <= key < len(self.names):
            self.names[key] = value.name
        if 0 <= key < len(self.scores):
            self.scores[key] = value.score
        if 0 <= key < len(self.pix_counts):
            self.pix_counts[key] = value.pix_count
        if 0 <= key < len(self.regions):
            self.regions[key] = value.region.__copy__()
        if 0 <= key < len(self.images):
            self.images[key] = value.image.__copy__()

    def clear(self):
        self.labels.clear()
        self.names.clear()
        self.scores.clear()
        self.pix_counts.clear()
        self.regions.clear()
        self.images.clear()

    def append(self, pObject: YoonObject):
        self.labels.append(pObject.label)
        self.names.append(pObject.name)
        self.scores.append(pObject.score)
        self.pix_counts.append(pObject.pix_count)
        self.regions.append(pObject.region.__copy__())
        self.images.append(pObject.image.__copy__())

    def min_size(self):
        nHeight = min([pImage.height for pImage in self.images])
        nWidth = min([pImage.width for pImage in self.images])
        return nWidth, nHeight

    def max_size(self):
        nHeight = max([pImage.height for pImage in self.images])
        nWidth = max([pImage.width for pImage in self.images])
        return nWidth, nHeight

    def max_channel(self):
        return max([pImage.channel for pImage in self.images])

    def min_channel(self):
        return min([pImage.channel for pImage in self.images])

    def resize(self,
               nWidth: int = 480,
               nHeight: int = 480,
               strOption=None  # "min", "max"
               ):
        if strOption == "min":
            nWidth, nHeight = self.min_size()
        elif strOption == "max":
            nWidth, nHeight = self.max_size()
        for iImage in range(len(self.images)):
            if isinstance(self.images[iImage], YoonImage):
                self.images[iImage] = self.images[iImage].resize(nWidth, nHeight)

    def rechannel(self,
                  nChannel: int = 1,
                  strOption=None  # "min", "max"
                  ):
        if strOption == "min":
            nChannel = self.min_channel()
        elif strOption == "max":
            nChannel = self.max_channel()
        for iImage in range(len(self.images)):
            if isinstance(self.images[iImage], YoonImage):
                self.images[iImage] = self.images[iImage].rechannel(nChannel)

    def normalize(self,
                  dMean: float = 128,
                  dStd: float = 255,
                  strOption=None  # "minmax", "z", "float"
                  ):
        if strOption == "minmax":
            for iImage in range(len(self.images)):
                if isinstance(self.images[iImage], YoonImage):
                    self.images[iImage] = self.images[iImage].minmax_normalize()[2]
        elif strOption == "z":
            for iImage in range(len(self.images)):
                if isinstance(self.images[iImage], YoonImage):
                    self.images[iImage] = self.images[iImage].z_normalize()[2]
        elif strOption == "float":
            for iImage in range(len(self.images)):
                if isinstance(self.images[iImage], YoonImage):
                    self.images[iImage] = self.images[iImage].normalize(0, 255)
        else:
            for iImage in range(len(self.images)):
                if isinstance(self.images[iImage], YoonImage):
                    self.images[iImage] = self.images[iImage].normalize(dMean, dStd)

    def denormalize(self,
                    dMean: float = 128,
                    dStd: float = 255):
        for iImage in range(len(self.images)):
            if isinstance(self.images[iImage], YoonImage):
                self.images[iImage] = self.images[iImage].denormalize(dMean, dStd)

    def to_region_points(self):
        pListResult = []
        for pRegion in self.regions:
            pListResult.append(pRegion.to_list())
        return pListResult

    def draw_dataset(self,
                     nRow: int = 4,
                     nCol: int = 4):
        nCountImage = len(self.images)
        nCountShow = nRow * nCol
        if nCountImage < nCountShow:
            print("!! Insufficient the count of images")
            return
        pFigure = matplotlib.pyplot.figure()
        pListRandom = numpy.random.randint(nCountImage, size=nCountShow)
        for i in range(nCountShow):
            pPlot = pFigure.add_subplot(nRow, nCol, i + 1)
            pPlot.set_xticks([])
            pPlot.set_yticks([])
            pImage = self.images[pListRandom[i]]
            nLabel = self.labels[pListRandom[i]]
            pPlot.set_title("{:3d}".format(nLabel))
            pPlot.imshow(pImage)
        matplotlib.pyplot.show()
