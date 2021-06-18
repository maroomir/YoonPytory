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


def get_phonemes():
    def to_dict():
        pDicPhn = {}
        for strTag in phoneme_list:
            if strTag.split(' ')[0] == 'q':
                pass
            else:
                pDicPhn[strTag.split(' ')[0]] = strTag.split(' ')[-1]
        return pDicPhn

    def to_list():
        pListPhn = [strTag.split(' ')[-1] for strTag in phoneme_list]
        pListPhn = list(set(pListPhn))
        return pListPhn

    g2p = G2p()
    pListPhoneme = ['h#']  # h# : start token
    pListPhoneme.extend(strPhoneme.lower() for strPhoneme in g2p(pListPhoneme))
    pListPhoneme.append('h#')  # h# : end token
    pListLabel = []
    for strLabel in pListPhoneme:
        if strLabel in ['q', ' ', "'"]:
            pass
        else:
            strLabel = ''.join([i for i in strLabel if not i.isdigit()])
            pListLabel.append(to_list().index(to_dict()[strLabel]) + 1)
    return numpy.concatenate(pListLabel)


class YoonObject(object):
    label = 0
    name = ""
    word = ""
    data_type = "deltas"
    speech: YoonSpeech = None

    def __init__(self,
                 nID: int = 0,
                 strName: str = "",
                 strWord: str = "",
                 strType: str = "deltas",
                 pSpeech: YoonSpeech = None):
        self.label = nID
        self.name = strName
        self.word = strWord
        self.data_type = strType
        if pSpeech is not None:
            self.speech = pSpeech.__copy__()

    def __copy__(self):
        return YoonObject(nID=self.label, strName=self.name, strType=self.data_type, pSpeech=self.speech)


class YoonDataset(object):
    labels: list = []
    names: list = []
    words: list = []
    data_types: list = []
    speeches: list = []

    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self.labels)

    def __init__(self,
                 pList: list = None,
                 *args: (YoonObject, YoonSpeech)):
        if len(args) > 0:
            iCount = 0
            for pItem in args:
                if isinstance(pItem, YoonObject):
                    self.labels.append(pItem.label)
                    self.names.append(pItem.name)
                    self.words.append(pItem.word)
                    self.data_types.append(pItem.data_type)
                    self.speeches.append(pItem.speech.__copy__())
                else:
                    self.labels.append(iCount)
                    self.data_types.append("deltas")
                    self.speeches.append(pItem.__copy__())
                iCount += 1
        else:
            if pList is not None:
                iCount = 0
                for pItem in args:
                    if isinstance(pItem, YoonObject):
                        self.labels.append(pItem.label)
                        self.names.append(pItem.name)
                        self.words.append(pItem.word)
                        self.data_types.append(pItem.data_type)
                        self.speeches.append(pItem.speech.__copy__())
                    elif isinstance(pItem, YoonSpeech):
                        self.labels.append(iCount)
                        self.data_types.append("deltas")
                        self.speeches.append(pItem.__copy__())
                    iCount += 1

    def __copy__(self):
        pResult = YoonDataset()
        pResult.labels = self.labels.copy()
        pResult.names = self.names.copy()
        pResult.speeches = self.speeches.copy()
        pResult.words = self.words.copy()
        return pResult

    def __getitem__(self, item: int):
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
                strWord: str = ""
                strType: str = "deltas"
                pSpeech: YoonSpeech = None
                if 0 <= item < len(self.labels):
                    nLabel = self.labels[item]
                if 0 <= item < len(self.names):
                    strName = self.names[item]
                if 0 <= item < len(self.words):
                    strWord = self.words[item]
                if 0 <= item < len(self.data_types):
                    strType = self.data_types[item]
                if 0 <= item < len(self.speeches):
                    pSpeech = self.speeches[item]
                pObject = YoonObject(nID=nLabel, strName=strName, pSpeech=pSpeech, strWord=strWord, strType=strType)
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
        if 0 <= key < len(self.data_types):
            self.data_types[key] = value.data_type
        if 0 <= key < len(self.words):
            self.words[key] = value.word
        if 0 <= key < len(self.speeches):
            self.speeches[key] = value.speech

    def clear(self):
        self.labels.clear()
        self.names.clear()
        self.words.clear()
        self.data_types.clear()
        self.speeches.clear()

    def append(self, pObject: YoonObject):
        self.labels.append(pObject.label)
        self.names.append(pObject.name)
        self.words.append(pObject.word)
        self.data_types.append(pObject.data_type)
        self.speeches.append(pObject.speech.__copy__())

    def to_features(self):
        pListFeature = []
        for iSpeech in range(len(self.speeches)):
            if isinstance(self.speeches[iSpeech], YoonSpeech):
                pListFeature.append(self.speeches[iSpeech].get_feature(self.data_types[iSpeech]))
        return numpy.array(pListFeature)

    def to_gmm_dataset(self):
        pArrayTarget = numpy.array(self.names)
        pArrayInput = self.to_features()
        return numpy.array(list(zip(pArrayInput, pArrayTarget)))

    def get_dimension(self):
        if len(self.speeches) == 0:
            raise Exception("The container size is zero")
        return self.speeches[0].get_dimension(self.data_types[0])
