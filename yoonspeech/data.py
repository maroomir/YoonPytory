import numpy
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
        phn_dic = {}
        for tag in phoneme_list:
            if tag.split(' ')[0] == 'q':
                pass
            else:
                phn_dic[tag.split(' ')[0]] = tag.split(' ')[-1]
        return phn_dic

    def to_list():
        phn_list = [strTag.split(' ')[-1] for strTag in phoneme_list]
        phn_list = list(set(phn_list))
        return phn_list

    g2p = G2p()
    phoneme_list = ['h#']  # h# : start token
    phoneme_list.extend(strPhoneme.lower() for strPhoneme in g2p(phoneme_list))
    phoneme_list.append('h#')  # h# : end token
    pListLabel = []
    for strLabel in phoneme_list:
        if strLabel in ['q', ' ', "'"]:
            pass
        else:
            strLabel = ''.join([i for i in strLabel if not i.isdigit()])
            pListLabel.append(to_list().index(to_dict()[strLabel]) + 1)
    return numpy.concatenate(pListLabel)


class YoonObject(object):
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    label = 0
    name = ""
    word = ""
    data_type = "deltas"
    speech: YoonSpeech = None
    """

    def __init__(self,
                 id: int = 0,
                 name: str = "",
                 word: str = "",
                 type: str = "deltas",
                 speech: YoonSpeech = None):
        self.label = id
        self.name = name
        self.word = word
        self.data_type = type
        self.speech = None if speech is None else speech.__copy__()

    def __copy__(self):
        return YoonObject(id=self.label, name=self.name, type=self.data_type, speech=self.speech)


class YoonDataset(object):
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    labels: list = []
    names: list = []
    words: list = []
    data_types: list = []
    speeches: list = []
    """

    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self.labels)

    def __init__(self):
        self.labels: list = []
        self.names: list = []
        self.words: list = []
        self.data_types: list = []
        self.speeches: list = []

    @classmethod
    def from_list(cls, data_list: list):
        dataset = YoonDataset()
        for i in range(len(data_list)):
            if isinstance(data_list[i], YoonObject):
                dataset.labels.append(data_list[i].label)
                dataset.names.append(data_list[i].name)
                dataset.words.append(data_list[i].word)
                dataset.data_types.append(data_list[i].data_type)
                dataset.speeches.append(data_list[i].speech.__copy__())
            elif isinstance(data_list[i], YoonSpeech):
                dataset.labels.append(i)
                dataset.data_types.append("deltas")
                dataset.speeches.append(data_list[i].__copy__())
        return dataset

    @classmethod
    def from_data(cls, *args: (YoonObject, YoonSpeech)):
        dataset = YoonDataset()
        for i in range(len(args)):
            if isinstance(args[i], YoonObject):
                dataset.labels.append(args[i].label)
                dataset.names.append(args[i].name)
                dataset.words.append(args[i].word)
                dataset.data_types.append(args[i].data_type)
                dataset.speeches.append(args[i].speech.__copy__())
            else:
                dataset.labels.append(i)
                dataset.data_types.append("deltas")
                dataset.speeches.append(args[i].__copy__())

    def __copy__(self):
        pResult = YoonDataset()
        pResult.labels = self.labels.copy()
        pResult.names = self.names.copy()
        pResult.speeches = self.speeches.copy()
        pResult.words = self.words.copy()
        return pResult

    def __getitem__(self, index):
        def get_object(i,
                       stop_index: int = None,
                       remains=None,
                       is_processing=False):
            if isinstance(i, tuple):
                index_list = list(i)[1:] + [None]
                result_list = []
                for start_index, remain_list in zip(i, index_list):
                    result, stop_index = get_object(start_index, stop_index, remain_list, True)
                    result_list.append(result)
                return result_list
            elif isinstance(i, slice):
                slice_range = range(*i.indices(len(self)))
                result_list = [get_object(j) for j in slice_range]
                if is_processing:
                    return result_list, slice_range[-1]
                else:
                    return result_list
            elif i is Ellipsis:
                if stop_index is not None:
                    stop_index += 1
                end_index = remains
                if isinstance(remains, slice):
                    end_index = remains.start
                result = get_object(slice(stop_index, end_index), is_processing=True)
                if is_processing:
                    return result[0], result[1]
                else:
                    return result[0]
            else:
                label: int = 0
                name: str = ""
                word: str = ""
                data_types: str = "deltas"
                speech: YoonSpeech
                if 0 <= i < len(self.labels):
                    label = self.labels[i]
                if 0 <= i < len(self.names):
                    name = self.names[i]
                if 0 <= i < len(self.words):
                    word = self.words[i]
                if 0 <= i < len(self.data_types):
                    data_types = self.data_types[i]
                if 0 <= i < len(self.speeches):
                    speech = self.speeches[i]
                pObject = YoonObject(id=label, name=name, speech=speech, word=word, type=data_types)
                if is_processing:
                    return pObject, i
                else:
                    return pObject

        selected_list = get_object(index)
        if isinstance(selected_list, list):
            return YoonDataset.from_list(selected_list)
        else:
            return selected_list

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
