import os
import numpy
import torch
from abc import ABCMeta, abstractmethod

from tqdm import tqdm
from numpy import ndarray
from torch.utils.data.dataset import T_co

from yoonimage.image import Image
from torch.utils.data import Dataset


class _Dataset(Dataset, metaclass=ABCMeta):
    def __init__(self, path: str):
        self.root_path = path
        self.inputs, self.targets = [], []
        self.params = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index) -> T_co:
        pass

    @abstractmethod
    def parse(self):
        pass


class ClassificationDataset(_Dataset):
    def __init__(self,
                 path: str = None,
                 images: list = None,
                 labels: list = None,
                 num_class: int = None):
        super().__init__(path)
        if isinstance(path, str):
            self.inputs, self.targets = self.parse()
        else:
            self.inputs, self.targets = images, labels
        self.inputs = numpy.array(self.inputs)
        self.targets = numpy.array(self.targets)
        self.output_dim = num_class

    @property
    def input_dim(self):
        return self.images.shape[-1]

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]

    def parse(self, root_path=None):
        if isinstance(root_path, str):
            self.root_path = root_path
        # Parse the file list
        images, labels = [], []
        for root_, dir_, path_ in tqdm(os.walk(self.root_path)):
            if len(path_) > 0:
                for pth_ in path_:
                    if os.path.splitext(pth_)[1] in [".jpg", ".bmp", ".png"]:
                        image = Image(path=os.path.join(root_, pth_))
                        images.append(image.tensor / 255.0)
                        labels.append(image.parents[-1])
        return images, labels


def parse_cifar_trainer(root: str, ratio: float = 0.8) -> dict:
    import pickle
    # Read the label names
    label = os.path.join(root, "batches.meta")
    with open(label, 'rb') as file:
        label_data = pickle.load(file)
        label_names = label_data['label_names']
    # Read the data
    path_ = [os.path.join(root, "data_batch_{}".format(i + 1)) for i in range(5)]
    images, labels = [], []
    for pth in path_:
        with open(pth, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            images.append(data[b'data'])
            labels.append(data[b'labels'])
    images = numpy.concatenate(images, axis=0)
    labels = numpy.concatenate(labels, axis=0)
    if images.shape[0] != labels.shape[0]:
        ValueError("The label and data size is not equal")
    # make the dataset
    cut_line = int(images.shape[0] * ratio)
    train_set, eval_set = [], []
    for i in range(cut_line):
        train_set += [{'image': images[i], 'label': labels[i]}]
    for i in range(cut_line, images.shape[0]):
        eval_set += [{'image': images[i], 'label': labels[i]}]
    print("Length of Train = {}".format(len(train_set)))
    print("Length of Test = {}".format(len(eval_set)))
    num_class = len(label_names)  # 10 (CIFAR-10)
    return {'train': train_set,
            'eval': eval_set,
            'num_class': num_class,
            'param': {
                'mean_norms': [0.4914, 0.4822, 0.4465],
                'std_norms': [0.247, 0.243, 0.261]
            }}


def parse_cifar_tester(root: str) -> dict:
    import pickle
    # Read the label names
    label_file = os.path.join(root, "batches.meta")
    with open(label_file, 'rb') as file:
        label_data = pickle.load(file)
        label_names = label_data['label_names']
    # Read the data
    path_ = os.path.join(root, "test_batch")
    with open(path_, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    datas = data['data']
    labels = data['label']
    if datas.shape[0] != labels.shape[0]:
        ValueError("The label and data size is not equal")
    # make the dataset
    test_set = []
    for i in range(datas.shape[0]):
        test_set += [{'image': datas[i], 'label': labels[i]}]
    print("Length of Test = {}".format(len(test_set)))
    num_class = label_names.shape[0]  # 10 (CIFAR-10)
    return {'test': test_set,
            'num_class': num_class,
            'param': {
                'mean_norms': [0.4914, 0.4822, 0.4465],
                'std_norms': [0.247, 0.243, 0.261]
            }}


HDF5_FORMAT = ['.h5', '.hdf5', '.mat']


def parse_hdf5_trainer(file_path: str,
                       input_lb: str = 'input',
                       target_lb: str = 'label',
                       ratio: float = 0.8) -> dict:
    assert os.path.splitext(file_path)[1] in HDF5_FORMAT, "Abnormal file format"
    import h5py
    # Read the hierarchical data file
    data = h5py.File(file_path)
    inputs = numpy.array(data[input_lb], dtype=numpy.float32)
    targets = numpy.array(data[target_lb], dtype=numpy.float32)
    # Transform data array to YoonDataset
    if inputs.shape[0] != targets.shape[0]:
        ValueError("The input and target data size is not equal")
    # make the dataset
    cut_line = int(inputs.shape[0] * ratio)
    train_set, eval_set = [], []
    for i in range(cut_line):
        train_set += [{'input': inputs[i], 'target': targets[i]}]
    for i in range(cut_line, inputs.shape[0]):
        eval_set += [{'input': inputs[i], 'target': targets[i]}]
    print("Length of Train = {}".format(len(train_set)))
    print("Length of Test = {}".format(len(eval_set)))
    # construct the dataset params
    return {'train': train_set, 'eval': eval_set}
