import numpy
import matplotlib.pyplot

from yoonpytory.figure import YoonLine2D, YoonRect2D, YoonVector2D
from yoonimage.image import YoonImage


class YoonObject:
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
                 region: (YoonVector2D, YoonLine2D, YoonRect2D) = None,
                 image: YoonImage = None):
        self.label = id_
        self.name = name
        self.score = score
        self.pixel_count = pix_count
        self.region = None if region is None else region.__copy__()
        self.image = None if image is None else image.__copy__()

    def __copy__(self):
        return YoonObject(id_=self.label, name=self.name, score=self.score, pix_count=self.pixel_count,
                          region=self.region, image=self.image)


class YoonDataset:
    # The shared area of YoonDataset class
    # All of instances are using this shared area
    # labels: list = []
    # names: list = []
    # scores: list = []
    # pix_counts: list = []
    # regions: list = []
    # images: list = []

    @staticmethod
    def from_tensor(images: numpy.ndarray,
                    labels: numpy.ndarray,
                    is_contain_batch: bool = True,
                    channel: int = 3):
        dataset = YoonDataset()
        count = len(images) if len(images) >= len(labels) else len(labels)
        step = 1 if is_contain_batch else channel
        for i in range(count, step):
            obj = YoonObject()
            obj.label = i if labels[i] is None else labels[i]
            if images[i] is not None:
                tensor = images[i] if is_contain_batch else numpy.concatenate([images[j + i] for j in range(channel)])
                obj.image = YoonImage.from_tensor(tensor)
            dataset.append(obj)
        return dataset

    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self._objects)

    def __init__(self):
        self._objects = []

    @classmethod
    def from_list(cls, list_: list):
        dataset = YoonDataset()
        for i in range(len(list_)):
            if isinstance(list_[i], YoonObject):
                dataset._objects.append(list_[i].__copy__())
            elif isinstance(list_[i], YoonImage):
                dataset._objects.append(YoonObject(id_=i, image=list_[i].__copy__()))
            elif isinstance(list_[i], (YoonRect2D, YoonLine2D, YoonVector2D)):
                dataset._objects.append(YoonObject(id_=i, region=list_[i].__copy__()))
        return dataset

    @classmethod
    def from_data(cls, *args: (YoonObject, YoonImage, YoonRect2D, YoonLine2D, YoonVector2D)):
        dataset = YoonDataset()
        for i in range(len(args)):
            if isinstance(args[i], YoonObject):
                dataset._objects.append(args[i].__copy__())
            elif isinstance(args[i], YoonImage):
                dataset._objects.append(YoonObject(id_=i, image=args[i].__copy__()))
            elif isinstance(args[i], (YoonRect2D, YoonLine2D, YoonVector2D)):
                dataset._objects.append(YoonObject(id_=i, region=args[i].__copy__()))
        return dataset

    def __copy__(self):
        pResult = YoonDataset()
        pResult._objects = self._objects.copy()
        return pResult

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
        return YoonDataset.from_list(result) if isinstance(result, list) else result

    def __setitem__(self, key: int, value: YoonObject):
        if 0 <= key < len(self):
            self._objects = value.__copy__()

    def clear(self):
        self._objects.clear()

    def append(self, obj: YoonObject):
        self._objects.append(obj)

    def images(self):
        return list(filter(None,
                           [self._objects[i].image for i in range(len(self))]))

    def labels(self):
        return list(filter(None,
                           [self._objects[i].label for i in range(len(self))]))

    def names(self):
        return list(filter(None,
                           [self._objects[i].name for i in range(len(self))]))

    def regions(self):
        return list(filter(None,
                           [self._objects[i].region for i in range(len(self))]))

    def min_size(self):
        images = self.images()
        height = min([_image.height for _image in images])
        width = min([_image.width for _image in images])
        return width, height

    def max_size(self):
        images = self.images()
        height = max([_image.height for _image in images])
        width = max([_image.width for _image in images])
        return width, height

    def max_channel(self):
        images = self.images()
        return max([_image.channel for _image in images])

    def min_channel(self):
        images = self.images()
        return min([_image.channel for _image in images])

    def to_region_points(self):
        regions = self.regions()
        return [_region.to_list() for _region in regions]

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


class YoonTransform(object):
    class Decimalize(object):
        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, YoonImage)
                result[i].image = dataset[i].image.pixel_decimal()
            return result

    class Recover(object):
        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, YoonImage)
                result[i].image = dataset[i].image.pixel_recover()
            return result

    class Normalization(object):
        def __init__(self,
                     mean_norms: list = None,
                     std_norms: list = None):
            self.means = mean_norms if isinstance(mean_norms, list) else [mean_norms]
            self.stds = std_norms if isinstance(std_norms, list) else [std_norms]
            assert len(self.means) == len(self.stds)
            assert 0 < len(self.means) <= 3

        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for channel in range(len(self.means)):
                mean = self.means[channel]
                std = self.stds[channel]
                for i in range(len(dataset)):
                    assert isinstance(dataset[i].image, YoonImage)
                    result[i].image = dataset[i].image.normalize(channel, mean=mean, std=std)
            return result

    class Denormalization(object):
        def __init__(self,
                     mean_norms: list = None,
                     std_norms: list = None):
            self.means = mean_norms if isinstance(mean_norms, list) else [mean_norms]
            self.stds = std_norms if isinstance(std_norms, list) else [std_norms]
            assert len(self.means) == len(self.stds)
            assert 0 < len(self.means) <= 3

        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for channel in range(len(self.means)):
                mean = self.means[channel]
                std = self.stds[channel]
                for i in range(len(dataset)):
                    assert isinstance(dataset[i].image, YoonImage)
                    result[i].image = dataset[i].image.denormalize(channel, mean=mean, std=std)
            return result

    class ZNormalization(object):
        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, YoonImage)
                result[i].image = dataset[i].image.z_normalize()[2]
            return result

    class MinMaxNormalization(object):
        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, YoonImage)
                result[i].image = dataset[i].image.minmax_normalize()[2]
            return result

    class Resize(object):
        def __init__(self,
                     width=0,
                     height=0,
                     is_padding=False):
            self.width = width
            self.height = height
            self.padding = is_padding

        def __call__(self, dataset: YoonDataset):
            if self.width == 0 or self.height == 0:
                self.width, self.height = dataset.min_size()
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, YoonImage)
                if self.padding:
                    result[i].image = dataset[i].image.resize_padding(self.width, self.height)
                else:
                    result[i].image = dataset[i].image.resize(self.width, self.height)
            return result

    class ResizeToMin(object):
        def __init__(self,
                     is_padding=False):
            self.padding = is_padding

        def __call__(self, dataset: YoonDataset):
            width, height = dataset.min_size()
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, YoonImage)
                if self.padding:
                    result[i].image = dataset[i].image.resize_padding(width, height)
                else:
                    result[i].image = dataset[i].image.resize(width, height)
            return result

    class Rechannel(object):
        def __init__(self,
                     channel=0):
            self.channel = channel

        def __call__(self, dataset: YoonDataset):
            if self.channel == 0:
                self.channel = dataset.min_channel()
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, YoonImage)
                result[i].image = dataset[i].image.rechannel(self.channel)
            return result

    class RechannelToMin(object):
        def __call__(self, dataset: YoonDataset):
            channel = dataset.min_channel()
            result = dataset.__copy__()
            for i in range(len(dataset)):
                assert isinstance(dataset[i].image, YoonImage)
                result[i].image = dataset[i].image.rechannel(channel)
            return result

    def __init__(self, *args):
        self.transforms = args

    def __call__(self, dataset: YoonDataset):
        for trans_func in self.transforms:
            dataset = trans_func(dataset)
        return dataset
