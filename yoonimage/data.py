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
                 id: int = 0,
                 name: str = "",
                 score=0.0,
                 pix_count: int = 0,
                 region: (YoonVector2D, YoonLine2D, YoonRect2D) = None,
                 image: YoonImage = None):
        self.label = id
        self.name = name
        self.score = score
        self.pixel_count = pix_count
        self.region = None if region is None else region.__copy__()
        self.image = None if image is None else image.__copy__()

    def __copy__(self):
        return YoonObject(id=self.label, name=self.name, score=self.score, pix_count=self.pixel_count,
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
                    labels: numpy.ndarray = None,
                    is_contain_batch: bool = True,
                    channel: int = 3):
        dataset = YoonDataset()
        if is_contain_batch:
            for i in range(len(images)):
                if labels is None:
                    dataset.labels.append(i)
                else:
                    dataset.labels.append(labels[i])
                if images is not None:
                    dataset.images.append(YoonImage.from_tensor(tensor=images[i]))
            return dataset
        else:
            for i in range(len(images), channel):
                if labels is None:
                    dataset.labels.append(i)
                else:
                    dataset.labels.append(labels[i])
                if images is not None:
                    dataset.images.append(YoonImage.from_tensor(tensor=numpy.concatenate([images[j + i]
                                                                                          for j in range(channel)])))
            return dataset

    def __str__(self):
        return "DATA COUNT {}".format(self.__len__())

    def __len__(self):
        return len(self.labels)

    def __init__(self):
        self.labels: list = []
        self.names: list = []
        self.scores: list = []
        self.pix_counts: list = []
        self.regions: list = []
        self.images: list = []

    @classmethod
    def from_list(cls, args: list):
        dataset = YoonDataset()
        i = 0
        for obj in args:
            if isinstance(obj, YoonObject):
                dataset.labels.append(obj.label)
                dataset.names.append(obj.name)
                dataset.scores.append(obj.score)
                dataset.pix_counts.append(obj.pixel_count)
                dataset.regions.append(obj.region.__copy__())
                dataset.images.append(obj.image.__copy__())
            elif isinstance(obj, YoonImage):
                dataset.labels.append(i)
                dataset.images.append(obj.__copy__())
            elif isinstance(obj, (YoonRect2D, YoonLine2D, YoonVector2D)):
                dataset.labels.append(i)
                dataset.regions.append(obj.__copy__())
            i += 1
        return dataset

    @classmethod
    def from_data(cls, *args: (YoonObject, YoonImage, YoonRect2D, YoonLine2D, YoonVector2D)):
        dataset = YoonDataset()
        i = 0
        for obj in args:
            if isinstance(obj, YoonObject):
                dataset.labels.append(obj.label)
                dataset.names.append(obj.name)
                dataset.scores.append(obj.score)
                dataset.pix_counts.append(obj.pixel_count)
                dataset.regions.append(obj.region.__copy__())
                dataset.images.append(obj.image.__copy__())
            elif isinstance(obj, YoonImage):
                dataset.labels.append(i)
                dataset.images.append(obj.__copy__())
            elif isinstance(obj, (YoonRect2D, YoonLine2D, YoonVector2D)):
                dataset.labels.append(i)
                dataset.regions.append(obj.__copy__())
            i += 1
        return dataset

    def __copy__(self):
        pResult = YoonDataset()
        pResult.labels = self.labels.copy()
        pResult.names = self.names.copy()
        pResult.scores = self.scores.copy()
        pResult.pix_counts = self.pix_counts.copy()
        pResult.regions = self.regions.copy()
        pResult.images = self.images.copy()
        return pResult

    def __getitem__(self, index):
        def get_object(i,
                       stop: int = None,
                       remains=None,
                       is_processing=False):
            if isinstance(i, tuple):
                index_list = list(i)[1:] + [None]
                result_list = []
                for start, remain_indexes in zip(i, index_list):
                    pResult, stop = get_object(start, stop, remain_indexes, True)
                    result_list.append(pResult)
                return result_list
            elif isinstance(i, slice):
                slice_range = range(*i.indices(len(self)))
                result_list = [get_object(j) for j in slice_range]
                if is_processing:
                    return result_list, slice_range[-1]
                else:
                    return result_list
            elif i is Ellipsis:
                if stop is not None:
                    stop += 1
                end_index = remains
                if isinstance(remains, slice):
                    end_index = remains.start
                pResult = get_object(slice(stop, end_index), is_processing=True)
                if is_processing:
                    return pResult[0], pResult[1]
                else:
                    return pResult[0]
            else:
                label: int = 0
                name: str = ""
                score: float = 0.00
                pix_count: int = 0
                region = None
                image: YoonImage
                if i > len(self):
                    raise IndexError("Index is too big")
                elif i < 0:
                    raise IndexError("Index is abnormal")
                if 0 <= i < len(self.labels):
                    label = self.labels[i]
                if 0 <= i < len(self.names):
                    name = self.names[i]
                if 0 <= i < len(self.scores):
                    score = self.scores[i]
                if 0 <= i < len(self.pix_counts):
                    pix_count = self.pix_counts[i]
                if 0 <= i < len(self.regions):
                    region = self.regions[i]
                if 0 <= i < len(self.images):
                    image = self.images[i]
                pObject = YoonObject(id=label, name=name, score=score, pix_count=pix_count,
                                     region=region, image=image)
                if is_processing:
                    return pObject, i
                else:
                    return pObject

        pResultItem = get_object(index)
        if isinstance(pResultItem, list):
            return YoonDataset.from_list(pResultItem)
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
            self.pix_counts[key] = value.pixel_count
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
        self.pix_counts.append(pObject.pixel_count)
        self.regions.append(pObject.region)
        self.images.append(pObject.image)

    def min_size(self):
        height = min([image.height for image in self.images])
        width = min([image.width for image in self.images])
        return width, height

    def max_size(self):
        height = max([image.height for image in self.images])
        width = max([image.width for image in self.images])
        return width, height

    def max_channel(self):
        return max([image.channel for image in self.images])

    def min_channel(self):
        return min([image.channel for image in self.images])

    def to_region_points(self):
        results = []
        for region in self.regions:
            results.append(region.to_list())
        return results

    def draw_dataset(self,
                     row: int = 4,
                     col: int = 4,
                     mode: str = "label"  # label, name
                     ):
        image_count = len(self.images)
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
            image = self.images[rand_list[i]].pixel_decimal().copy_buffer()
            if mode == "label":
                label = self.labels[rand_list[i]]
                plot.set_title("{:3d}".format(label))
            elif mode == "name":
                name = self.names[rand_list[i]]
                plot.set_title("{}".format(name))
            plot.imshow(image)
        matplotlib.pyplot.show()


class YoonTransform(object):
    class Decimalize(object):
        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for iImage in range(len(dataset.images)):
                assert isinstance(dataset.images[iImage], YoonImage)
                result.images[iImage] = dataset.images[iImage].pixel_decimal()
            return result

    class Recover(object):
        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for iImage in range(len(dataset.images)):
                assert isinstance(dataset.images[iImage], YoonImage)
                result.images[iImage] = dataset.images[iImage].pixel_recover()
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
                for image in range(len(dataset.images)):
                    assert isinstance(dataset.images[image], YoonImage)
                    result.images[image] = dataset.images[image].normalize(channel, mean=mean, std=std)
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
                for image in range(len(dataset.images)):
                    assert isinstance(dataset.images[image], YoonImage)
                    result.images[image] = dataset.images[image].denormalize(channel, mean=mean, std=std)
            return result

    class ZNormalization(object):
        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for iImage in range(len(dataset.images)):
                assert isinstance(dataset.images[iImage], YoonImage)
                result.images[iImage] = dataset.images[iImage].z_normalize()[2]
            return result

    class MinMaxNormalization(object):
        def __call__(self, dataset: YoonDataset):
            result = dataset.__copy__()
            for image in range(len(dataset.images)):
                assert isinstance(dataset.images[image], YoonImage)
                result.images[image] = dataset.images[image].minmax_normalize()[2]
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
            for i in range(len(dataset.images)):
                assert isinstance(dataset.images[i], YoonImage)
                if self.padding:
                    result.images[i] = dataset.images[i].resize_padding(self.width, self.height)
                else:
                    result.images[i] = dataset.images[i].resize(self.width, self.height)
            return result

    class ResizeToMin(object):
        def __init__(self,
                     is_padding=False):
            self.padding = is_padding

        def __call__(self, dataset: YoonDataset):
            width, height = dataset.min_size()
            result = dataset.__copy__()
            for i in range(len(dataset.images)):
                assert isinstance(dataset.images[i], YoonImage)
                if self.padding:
                    result.images[i] = dataset.images[i].resize_padding(width, height)
                else:
                    result.images[i] = dataset.images[i].resize(width, height)
            return result

    class Rechannel(object):
        def __init__(self,
                     channel=0):
            self.channel = channel

        def __call__(self, dataset: YoonDataset):
            if self.channel == 0:
                self.channel = dataset.min_channel()
            result = dataset.__copy__()
            for i in range(len(dataset.images)):
                assert isinstance(dataset.images[i], YoonImage)
                result.images[i] = dataset.images[i].rechannel(self.channel)
            return result

    class RechannelToMin(object):
        def __call__(self, dataset: YoonDataset):
            channel = dataset.min_channel()
            result = dataset.__copy__()
            for i in range(len(dataset.images)):
                assert isinstance(dataset.images[i], YoonImage)
                result.images[i] = dataset.images[i].rechannel(channel)
            return result

    def __init__(self, *args):
        self.transforms = args

    def __call__(self, dataset: YoonDataset):
        for trans_func in self.transforms:
            dataset = trans_func(dataset)
        return dataset
