import numpy

import yoonspeech
from g2p_en import G2p
from yoonspeech.speech import YoonSpeech

phoneme_list = ["aa aa aa", "ae ae ae", "ah ah ah", "ao ao aa", "aw aw aw", "ax ax ah", "ax-h ax ah", "axr er er",  # 8
                "ay ay ay",  # 1
                "b b b", "bcl vcl sil",  # 2
                "ch ch ch",  # 1
                "d d d", "dcl vcl sil", "dh dh dh", "dx dx dx",  # 4
                "eh eh eh", "el el l", "em m m", "en en n", "eng ng ng", "epi epi sil", "er er er", "ey ey ey",  # 8
                "f f f",  # 1
                "g g g", "gcl vcl sil",  # 2
                "h# sil sil", "hh hh hh", "hv hh hh",  # 3
                "ih ih ih", "ix ix ih", "iy iy iy",  # 3
                "jh jh jh",  # 1
                "k  k  k", "kcl cl sil",  # 2
                "l l l",  # 1
                "m m m",  # 1
                "n n n", "ng ng ng", "nx n n",  # 3
                "ow ow ow", "oy oy oy",  # 2
                "p p p", "pau sil sil", "pcl cl sil",  # 3
                "q",  # 1
                "r r r",  # 1
                "s s s", "sh sh sh",  # 2
                "t t t", "tcl cl sil", "th th th",  # 3
                "uh uh uh", "uw uw uw", "ux uw uw",  # 3
                "v v v",  # 1
                "w w w",  # 1
                "y y y",  # 1
                "z z z", "zh zh sh",  # 2
                "sil sil sil"]  # 1

def get_phoneme_dict():
    pDicPhn = {}
    for strTag in phoneme_list:
        if strTag.split(' ')[0] == 'q':
            pass
        else:
            pDicPhn[strTag.split(' ')[0]] = strTag.split(' ')[-1]
    return pDicPhn


def get_phoneme_list():
    pListPhn = [strTag.split(' ')[-1] for strTag in phoneme_list]
    pListPhn = list(set(pListPhn))
    return pListPhn


class YoonObject(object):
    label = 0
    name = ""
    word = ""
    speech: YoonSpeech = None
    buffer: numpy.ndarray = None
    __g2p = G2p()
    __phoneme_dict = get_phoneme_dict()
    __phoneme_list = get_phoneme_list()

    def __init__(self,
                 nID: int = 0,
                 strName: str = "",
                 strWord: str = "",
                 pSpeech: YoonSpeech = None,
                 pBuffer: numpy.ndarray = None,
                 strType: str = "mfcc"):
        self.label = nID
        self.name = strName
        self.word = strWord
        if pSpeech is not None:
            self.speech = pSpeech.__copy__()
            self.buffer = pSpeech.get_feature(strFeatureType=strType)
        if pBuffer is not None:
            self.buffer = pBuffer.copy()

    def __copy__(self):
        return YoonObject(nID=self.label, strName=self.name, pSpeech=self.speech, pBuffer=self.buffer)

    def get_dimension(self):
        if self.speech is not None:
            return self.speech.get_dimension()
        else:
            Exception("Speech object is null")

    def get_phonemes(self):
        pListPhoneme = ['h#']  # h# : start token
        pListPhoneme.extend(strPhoneme.lower() for strPhoneme in self.__g2p(pListPhoneme))
        pListPhoneme.append('h#')  # h# : end token
        pListLabel = []
        for strLabel in pListPhoneme:
            if strLabel in ['q', ' ', "'"]:
                pass
            else:
                strLabel = ''.join([i for i in strLabel if not i.isdigit()])
                pListLabel.append(self.__phoneme_list.index(self.__phoneme_dict[strLabel]) + 1)
        return pListLabel

    def get_phonemes_array(self):
        return numpy.array(self.get_phonemes())


class YoonDataset(object):
    speaker_count: int = 0
    phoneme_count: int = 0
    labels: list = []
    names: list = []
    words: list = []
    speechs: list = []
    buffers: list = []
    __buffer_type: str = "mfcc"

    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self.labels)

    def __init__(self,
                 nSpeakers: int,
                 nPhonemes: int,
                 strType: str = "mfcc",
                 pList: list = None,
                 *args: (YoonObject, YoonSpeech)):
        self.speaker_count = nSpeakers
        self.phoneme_count = nPhonemes
        self.__buffer_type = strType
        if len(args) > 0:
            iCount = 0
            for pItem in args:
                if isinstance(pItem, YoonObject):
                    self.labels.append(pItem.label)
                    self.names.append(pItem.name)
                    self.speechs.append(pItem.speech.__copy__())
                    self.words.append(pItem.word)
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
                        self.words.append(pItem.word)
                        self.buffers.append(pItem.buffer)
                    elif isinstance(pItem, YoonSpeech):
                        self.labels.append(iCount)
                        self.speechs.append(pItem.__copy__())
                        self.buffers.append(pItem.get_feature(strFeatureType=strType))
                    iCount += 1

    def __copy__(self):
        pResult = YoonDataset(self.class_count)
        pResult.labels = self.labels.copy()
        pResult.names = self.names.copy()
        pResult.speechs = self.speechs.copy()
        pResult.words = self.words.copy()
        pResult.buffers = self.buffers.copy()
        pResult.__buffer_type = self.__buffer_type
        return pResult

    def __getitem__(self, item: int):
        nLabel: int = 0
        strName: str = ""
        strWord: str = ""
        pSpeech: YoonSpeech = None
        pBuffer: numpy.ndarray = None
        if 0 <= item < len(self.labels):
            nLabel = self.labels[item]
        if 0 <= item < len(self.names):
            strName = self.names[item]
        if 0 <= item < len(self.speechs):
            pSpeech = self.speechs[item]
        if 0 <= item < len(self.words):
            strWord = self.words[item]
        if 0 <= item < len(self.buffers):
            pBuffer = self.buffers[item]
        return YoonObject(nID=nLabel, strName=strName, pSpeech=pSpeech, pBuffer=pBuffer, strWord=strWord,
                          strType=self.__buffer_type)

    def __setitem__(self, key: int, value: YoonObject):
        if 0 <= key < len(self.labels):
            self.labels[key] = value.label
        if 0 <= key < len(self.names):
            self.names[key] = value.name
        if 0 <= key < len(self.speechs):
            self.speechs[key] = value.speech
        if 0 <= key < len(self.buffers):
            self.buffers[key] = value.buffer
        if 0 <= key < len(self.words):
            self.words[key] = value.word

    def clear(self):
        self.labels.clear()
        self.names.clear()
        self.speechs.clear()
        self.words.clear()
        self.buffers.clear()

    def append(self, pObject: YoonObject):
        self.labels.append(pObject.label)
        self.names.append(pObject.name)
        self.speechs.append(pObject.speech.__copy__())
        self.words.append(pObject.word)
        self.buffers.append(pObject.buffer.copy())

    def to_gmm_set(self):
        pArrayTarget = numpy.array(self.names)
        pArrayInput = numpy.array(self.buffers)
        return numpy.array(list(zip(pArrayInput, pArrayTarget)))

    def get_dimension(self):
        if len(self.speechs) == 0:
            raise Exception("The container size is zero")
        return self.speechs[0].get_dimension()
