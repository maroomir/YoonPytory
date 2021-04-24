import numpy

from yoonspeech.speech import YoonSpeech


class YoonObject(object):
    label = 0
    name = ""
    speech: YoonSpeech = None
    buffer: numpy.ndarray = None
    __bufferType: str = "mfcc"

    def __init__(self,
                 nID: int = 0,
                 strName: str = "",
                 pSpeech: YoonSpeech = None,
                 pBuffer: numpy.ndarray = None,
                 strType: str = "mfcc"):
        self.label = nID
        self.name = strName
        self.__bufferType = strType
        if pSpeech is not None:
            self.speech = pSpeech.__copy__()
            self.buffer = pSpeech.get_feature(strFeatureType=strType)
        if pBuffer is not None:
            self.buffer = pBuffer.copy()

    def __copy__(self):
        return YoonObject(nID=self.label, strName=self.name, pSpeech=self.speech, pBuffer=self.buffer)

    def get_dimension(self):
        if self.speech is not None:
            return self.speech.get_dimension(self.__bufferType)
        else:
            Exception("Speech object is null")


class YoonDataset(object):
    classificationCount: int = 0
    labels: list = []
    names: list = []
    speechs: list = []
    buffers: list = []
    __bufferType: str = "mfcc"

    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self.labels)

    def __init__(self,
                 nCount: int,
                 pList: list = None,
                 strType: str = "mfcc",
                 *args: (YoonObject, YoonSpeech)):
        self.classificationCount = nCount
        self.__bufferType = strType
        if len(args) > 0:
            iCount = 0
            for pItem in args:
                if isinstance(pItem, YoonObject):
                    self.labels.append(pItem.label)
                    self.names.append(pItem.name)
                    self.speechs.append(pItem.speech.__copy__())
                    self.buffers.append(pItem.buffer)
                else:
                    self.labels.append(iCount)
                    self.speechs.append(pItem.__copy__())
                    self.buffers.append(pItem.get_feature(strFeatureType=strType))
                iCount += 1
        else:
            if pList is not None:
                iCount = 0
                for pItem in args:
                    if isinstance(pItem, YoonObject):
                        self.labels.append(pItem.label)
                        self.names.append(pItem.name)
                        self.speechs.append(pItem.speech.__copy__())
                        self.buffers.append(pItem.buffer)
                    elif isinstance(pItem, YoonSpeech):
                        self.labels.append(iCount)
                        self.speechs.append(pItem.__copy__())
                        self.buffers.append(pItem.get_feature(strFeatureType=strType))
                    iCount += 1

    def __copy__(self):
        pResult = YoonDataset(self.classificationCount)
        pResult.labels = self.labels.copy()
        pResult.names = self.names.copy()
        pResult.speechs = self.speechs.copy()
        pResult.buffers = self.buffers.copy()
        pResult.__bufferType = self.__bufferType
        return pResult

    def __getitem__(self, item: int):
        nLabel: int = 0
        strName: str = ""
        pSpeech: YoonSpeech = None
        pBuffer: numpy.ndarray = None
        if 0 <= item < len(self.labels):
            nLabel = self.labels[item]
        if 0 <= item < len(self.names):
            strName = self.names[item]
        if 0 <= item < len(self.speechs):
            pSpeech = self.speechs[item]
        if 0 <= item < len(self.buffers):
            pBuffer = self.buffers[item]
        return YoonObject(nID=nLabel, strName=strName, pSpeech=pSpeech, pBuffer=pBuffer,
                          strType=self.__bufferType)

    def __setitem__(self, key: int, value: YoonObject):
        if 0 <= key < len(self.labels):
            self.labels[key] = value.label
        if 0 <= key < len(self.names):
            self.names[key] = value.name
        if 0 <= key < len(self.speechs):
            self.speechs[key] = value.speech
        if 0 <= key < len(self.buffers):
            self.buffers[key] = value.buffer

    def clear(self):
        self.labels.clear()
        self.names.clear()
        self.speechs.clear()
        self.buffers.clear()

    def append(self, pObject: YoonObject):
        self.labels.append(pObject.label)
        self.names.append(pObject.name)
        self.speechs.append(pObject.speech.__copy__())
        self.buffers.append(pObject.buffer.copy())

    def to_dataset(self):
        pArrayNames = numpy.array(self.names)
        pArrayData = numpy.array(self.buffers)
        return numpy.array(list(zip(pArrayNames, pArrayData)))
