import abc
from abc import ABCMeta, abstractmethod

import numpy
import matplotlib.pyplot
from numpy import ndarray

from yoonpytory.figure import Line2D, Rect2D, Vector2D
from yoonimage.image import Image


class Object:
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    label = 0
    name = ""
    score = 0.0
    pix_count = 0
    region: (YoonRect2D, YoonLine2D, YoonVector2D) = None
    image: YoonImage = None
    """

    def __init__(self,
                 id_: int = 0,
                 name: str = "",
                 score=0.0,
                 pix_count: int = 0,
                 pos: Vector2D = None,
                 region: (Line2D, Rect2D) = None,
                 image: Image = None):
        pos = region.feature_pos if pos is None and region is not None else pos
        self.label = id_
        self.name = name
        self.score = score
        self.pixel_count = pix_count
        self.pos = None if pos is None else pos.__copy__()
        self.region = None if region is None else region.__copy__()
        self.image = None if image is None else image.__copy__()

    def __copy__(self):
        return Object(id_=self.label, name=self.name, score=self.score, pix_count=self.pixel_count,
                      region=self.region, image=self.image)


class _Dataset(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def from_tensor(images: numpy.ndarray, labels: numpy.ndarray):
        pass

    @abstractmethod
    def __copy__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def draw_dataset(self, row: int, col: int):
        pass

    @property
    @abstractmethod
    def images(self):
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

    @property
    @abstractmethod
    def min_size(self):
        pass

    @property
    @abstractmethod
    def max_size(self):
        pass

    @property
    @abstractmethod
    def max_channel(self):
        pass

    @property
    @abstractmethod
    def min_channel(self):
        pass


class Dataset1D(_Dataset):
    """
    # The shared area of YoonDataset class
    # All of instances are using this shared area
    # labels: list = []
    # names: list = []
    # scores: list = []
    # pix_counts: list = []
    # regions: list = []
    # images: list = []
    """

    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self._objects)

    def __init__(self):
        self._objects = []

    @staticmethod
    def from_tensor(images: numpy.ndarray,
                    labels: numpy.ndarray,
                    contain_batch: bool = True,
                    channel: int = 3):
        dataset = Dataset1D()
        count = len(images) if len(images) >= len(labels) else len(labels)
        step = 1 if contain_batch else channel
        for i in range(count, step):
            obj = Object()
            obj.label = i if labels[i] is None else labels[i]
            if images[i] is not None:
                tensor = images[i] if contain_batch else numpy.concatenate([images[j + i] for j in range(channel)])
                obj.image = Image.from_tensor(tensor)
            dataset.append(obj)
        return dataset

    @classmethod
    def from_list(cls, list_: list):
        dataset = Dataset1D()
        for i in range(len(list_)):
            if isinstance(list_[i], Object):
                dataset._objects.append(list_[i].__copy__())
            elif isinstance(list_[i], Image):
                dataset._objects.append(Object(id_=i, image=list_[i].__copy__()))
            elif isinstance(list_[i], (Rect2D, Line2D)):
                dataset._objects.append(Object(id_=i, region=list_[i].__copy__()))
            elif isinstance(list_[i], Vector2D):
                dataset._objects.append(Object(id_=i, pos=list_[i].__copy__()))
        return dataset

    @classmethod
    def from_data(cls, *args: (Object, Image, Rect2D, Line2D, Vector2D)):
        dataset = Dataset1D()
        for i in range(len(args)):
            if isinstance(args[i], Object):
                dataset._objects.append(args[i].__copy__())
            elif isinstance(args[i], Image):
                dataset._objects.append(Object(id_=i, image=args[i].__copy__()))
            elif isinstance(args[i], (Rect2D, Line2D)):
                dataset._objects.append(Object(id_=i, region=args[i].__copy__()))
            elif isinstance(args[i], Vector2D):
                dataset._objects.append(Object(id_=i, pos=args[i].__copy__()))
        return dataset

    @classmethod
    def from_feature_list(cls, list_: list):
        dataset = Dataset1D()
        for feature in list_:
            if len(feature) == 2:
                obj_ = Object(pos=Vector2D.from_array(feature))
            elif len(feature) == 4:
                obj_ = Object(region=Rect2D.from_array(feature))
            else:
                continue
            dataset.append(obj_)

    @classmethod
    def from_feature_array(cls, array: ndarray):
        dataset = Dataset1D()
        assert len(array.shape) == 2
        for i in array.shape[0]:
            if array.shape[1] == 2:
                obj_ = Object(pos=Vector2D.from_array(array[i]))
            elif array.shape[1] == 4:
                obj_ = Object(region=Rect2D.from_array(array[i]))
            else:
                continue
            dataset.append(obj_)
        return dataset

    def __copy__(self):
        result = Dataset1D()
        result._objects = self._objects.copy()
        return result

    def __getitem__(self, index):
        def get_object(i,
                       stop: int = None,
                       remains=None,
                       is_processing=False):
            if isinstance(i, tuple):
                indexes = list(i)[1:] + [None]
                results = []
                for _start, _remains in zip(i, indexes):
                    _result, stop = get_object(_start, stop, _remains, True)
                    results.append(_result)
                return results
            elif isinstance(i, slice):
                range_ = range(*i.indices(len(self)))
                results = [get_object(j) for j in range_]
                if is_processing:
                    return results, range_[-1]
                else:
                    return results
            elif i is Ellipsis:
                stop += (1 if stop is not None else 0)
                end_ = remains
                if isinstance(remains, slice):
                    end_ = remains.start
                _result = get_object(slice(stop, end_), is_processing=True)
                if is_processing:
                    return _result[0], _result[1]
                else:
                    return _result[0]
            else:
                if i > len(self):
                    raise IndexError("Index is too big")
                elif i < 0:
                    raise IndexError("Index is abnormal")
                if is_processing:
                    return self._objects[i], i
                else:
                    return self._objects[i]

        result = get_object(index)
        return Dataset1D.from_list(result) if isinstance(result, list) else result

    def __setitem__(self, key: int, value: Object):
        if 0 <= key < len(self):
            self._objects = value.__copy__()

    def clear(self):
        self._objects.clear()

    def append(self, obj: Object):
        self._objects.append(obj)

    @property
    def images(self):
        return list(filter(None,
                           [self._objects[i].image for i in range(len(self))]))

    @property
    def labels(self):
        return list(filter(None,
                           [self._objects[i].label for i in range(len(self))]))

    @property
    def names(self):
        return list(filter(None,
                           [self._objects[i].name for i in range(len(self))]))

    @property
    def regions(self):
        return list(filter(None,
                           [self._objects[i].region for i in range(len(self))]))

    @property
    def poses(self):
        return list(filter(None,
                           [self._objects[i].pos for i in range(len(self))]))

    @property
    def min_size(self):
        height = min([_image.height for _image in self.images])
        width = min([_image.width for _image in self.images])
        return width, height

    @property
    def max_size(self):
        height = max([_image.height for _image in self.images])
        width = max([_image.width for _image in self.images])
        return width, height

    @property
    def max_channel(self):
        return max([_image.channel for _image in self.images])

    @property
    def min_channel(self):
        return min([_image.channel for _image in self.images])

    @property
    def region_points(self):
        return [_region.to_list() for _region in self.regions]

    def draw_dataset(self,
                     row: int = 4,
                     col: int = 4,
                     mode: str = "label"  # label, name
                     ):
        image_count = len(self)
        show_count = row * col
        if image_count < show_count:
            print("!! Insufficient the count of images")
            return
        figure = matplotlib.pyplot.figure()
        rand_list = numpy.random.randint(image_count, size=show_count)
        for i in range(show_count):
            plot = figure.add_subplot(row, col, i + 1)
            plot.set_xticks([])
            plot.set_yticks([])
            image = self._objects[rand_list[i]].image.pixel_decimal().copy_buffer()
            if mode == "label":
                label = self._objects[rand_list[i]].label
                plot.set_title("{:3d}".format(label))
            elif mode == "name":
                name = self._objects[rand_list[i]].name
                plot.set_title("{}".format(name))
            plot.imshow(image)
        matplotlib.pyplot.show()


class Dataset2D(_Dataset):
    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self._images)

    def __init__(self):
        self._images = []
        self._labels = []

    @staticmethod
    def from_tensor(images: numpy.ndarray,
                    labels: numpy.ndarray):
        dataset = Dataset2D()
        for img, lb in zip(images, labels):
            dataset.append(img, lb)
        return dataset

    @classmethod
    def from_list(cls, images: list, labels: list):
        dataset = Dataset2D()
        for img, lb in zip(images, labels):
            dataset.append(img, lb)
        return dataset

    def __copy__(self):
        result = Dataset2D()
        result._images = self._images.copy()
        result._labels = self._labels.copy()

    def __getitem__(self, index):
        def get_object(i,
                       stop: int = None,
                       remains=None,
                       is_processing=False):
            if isinstance(i, tuple):
                indexes = list(i)[1:] + [None]
                img_results, lb_results = [], []
                for _start, _remains in zip(i, indexes):
                    _image, _label, stop = get_object(_start, stop, _remains, True)
                    img_results.append(_image), lb_results.append(_label)
                return img_results, lb_results
            elif isinstance(i, slice):
                range_ = range(*i.indices(len(self)))
                img_results, lb_results = [], []
                for j in range_:
                    _image, _label = get_object(j)
                    img_results.append(_image), lb_results.append(_label)
                if is_processing:
                    return img_results, lb_results, range_[-1]
                else:
                    return img_results, lb_results
            elif i is Ellipsis:
                stop += (1 if stop is not None else 0)
                end_ = remains
                if isinstance(remains, slice):
                    end_ = remains.start
                _image, _label, stop = get_object(slice(stop, end_), is_processing=True)
                if is_processing:
                    return _image, _label, stop
                else:
                    return _image, _label
            else:
                if i > len(self):
                    raise IndexError("Index is too big")
                elif i < 0:
                    raise IndexError("Index is abnormal")
                if is_processing:
                    return self._images[i], self._labels[i], i
                else:
                    return self._images[i], self._labels[i]

        images, labels = get_object(index)
        return Dataset2D.from_list(images, labels) \
                   if isinstance(images, list) and isinstance(labels, list) else images, labels

    def __setitem__(self, key: int, value: tuple):
        if 0 <= key < len(self):
            self._images = value[0].__copy__()
            self._labels = value[1].__copy__()

    def clear(self):
        self._images.clear()
        self._labels.clear()

    def append(self, image: Image, label: Image):
        self._images.append(image)
        self._labels.append(label)

    def draw_dataset(self, row: int, col: int):
        pass

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def min_size(self):
        height = min([_image.height for _image in self.images] + [_label.height for _label in self.labels])
        width = min([_image.width for _image in self.images] + [_label.width for _label in self.labels])
        return width, height

    @property
    def max_size(self):
        height = max([_image.height for _image in self.images] + [_label.height for _label in self.labels])
        width = max([_image.width for _image in self.images] + [_label.width for _label in self.labels])
        return width, height

    @property
    def max_channel(self):
        return max([_image.channel for _image in self.images] + [_label.channel for _label in self.labels])

    @property
    def min_channel(self):
        return min([_image.channel for _image in self.images] + [_label.channel for _label in self.labels])


class _Transform:
    @abstractmethod
    def __call__(self, dataset: _Dataset):
        pass


class Transform1D(_Transform):
    class Decimalize(_Transform):
        def __call__(self, dataset: Dataset1D):
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, Image)
                result[i].image = dataset[i].image.pixel_decimal()
            return result

    class Recover(_Transform):
        def __call__(self, dataset: Dataset1D):
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, Image)
                result[i].image = dataset[i].image.pixel_recover()
            return result

    class Normalization(_Transform):
        def __init__(self,
                     mean_norms: list = None,
                     std_norms: list = None):
            self.means = mean_norms if isinstance(mean_norms, list) else [mean_norms]
            self.stds = std_norms if isinstance(std_norms, list) else [std_norms]
            assert len(self.means) == len(self.stds)
            assert 0 < len(self.means) <= 3

        def __call__(self, dataset: Dataset1D):
            result = dataset.__copy__()
            for channel in range(len(self.means)):
                mean = self.means[channel]
                std = self.stds[channel]
                for i in range(len(dataset)):
                    assert isinstance(dataset[i].image, Image)
                    result[i].image = dataset[i].image.normalize(channel, mean=mean, std=std)
            return result

    class Denormalization(_Transform):
        def __init__(self,
                     mean_norms: list = None,
                     std_norms: list = None):
            self.means = mean_norms if isinstance(mean_norms, list) else [mean_norms]
            self.stds = std_norms if isinstance(std_norms, list) else [std_norms]
            assert len(self.means) == len(self.stds)
            assert 0 < len(self.means) <= 3

        def __call__(self, dataset: Dataset1D):
            result = dataset.__copy__()
            for channel in range(len(self.means)):
                mean = self.means[channel]
                std = self.stds[channel]
                for i in range(len(dataset)):
                    assert isinstance(dataset[i].image, Image)
                    result[i].image = dataset[i].image.denormalize(channel, mean=mean, std=std)
            return result

    class ZNormalization(_Transform):
        def __call__(self, dataset: Dataset1D):
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, Image)
                result[i].image = dataset[i].image.z_normalize()[2]
            return result

    class MinMaxNormalization(_Transform):
        def __call__(self, dataset: Dataset1D):
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, Image)
                result[i].image = dataset[i].image.minmax_normalize()[2]
            return result

    class Resize(_Transform):
        def __init__(self,
                     width=0,
                     height=0,
                     is_padding=False):
            self.width = width
            self.height = height
            self.padding = is_padding

        def __call__(self, dataset: Dataset1D):
            if self.width == 0 or self.height == 0:
                self.width, self.height = dataset.min_size
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, Image)
                if self.padding:
                    result[i].image = dataset[i].image.resize_padding(self.width, self.height)
                else:
                    result[i].image = dataset[i].image.resize(self.width, self.height)
            return result

    class ResizeToMin(_Transform):
        def __init__(self,
                     is_padding=False):
            self.padding = is_padding

        def __call__(self, dataset: Dataset1D):
            width, height = dataset.min_size
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, Image)
                if self.padding:
                    result[i].image = dataset[i].image.resize_padding(width, height)
                else:
                    result[i].image = dataset[i].image.resize(width, height)
            return result

    class Rechannel(_Transform):
        def __init__(self,
                     channel=0):
            self.channel = channel

        def __call__(self, dataset: Dataset1D):
            if self.channel == 0:
                self.channel = dataset.min_channel()
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, Image)
                result[i].image = dataset[i].image.rechannel(self.channel)
            return result

    class RechannelToMin(_Transform):
        def __call__(self, dataset: Dataset1D):
            channel = dataset.min_channel()
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, Image)
                result[i].image = dataset[i].image.rechannel(channel)
            return result

    def __init__(self, *args):
        self.transforms = args

    def __call__(self, dataset: Dataset1D):
        for trans_func in self.transforms:
            dataset = trans_func(dataset)
        return dataset
