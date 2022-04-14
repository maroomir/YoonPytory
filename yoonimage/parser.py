import os
import numpy

from yoonimage.data import Dataset1D, Object, Transform1D
from yoonimage.image import Image


def parse_root(root: str):
    # Parse the file list
    paths = os.listdir(root)
    dataset = Dataset1D()
    count = 0
    for path in paths:
        if "jpg" in path or "bmp" in path or "png" in path:
            image_path = os.path.join(root, path)
            image = Image.from_path(image_path)
            obj = Object(id_=count, name=path, image=image)
            dataset.append(obj)
            count += 1
    return count, dataset


def parse_cifar10_trainer(root: str,
                          train_ratio: float = 0.8,
                          ):
    import pickle
    # Read the label names
    label_path = os.path.join(root, "batches.meta")
    with open(label_path, 'rb') as file:
        label_data = pickle.load(file)
        label_names = label_data['label_names']
    # Read the data
    train_paths = [os.path.join(root, "data_batch_{}".format(i + 1)) for i in range(5)]
    datas = []
    labels = []
    for path in train_paths:
        with open(path, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            datas.append(data[b'data'])
            labels.append(data[b'labels'])
    datas = numpy.concatenate(datas, axis=0)
    labels = numpy.concatenate(labels, axis=0)
    # Transform data array to YoonDataset
    if datas.shape[0] != labels.shape[0]:
        ValueError("The label and data size is not equal")
    train_dataset = Dataset1D()
    eval_dataset = Dataset1D()
    cut_line = int(datas.shape[0] * train_ratio)
    for i in range(cut_line):
        image = Image.parse_array(32, 32, 3, datas[i], mode="parallel")
        label = labels[i]
        obj = Object(id_=label, name=label_names[label], image=image)
        train_dataset.append(obj)
    for i in range(cut_line, datas.shape[0]):
        image = Image.parse_array(32, 32, 3, datas[i], mode="parallel")
        label = labels[i]
        obj = Object(id_=label, name=label_names[label], image=image)
        eval_dataset.append(obj)
    print("Length of Train = {}".format(train_dataset.__len__()))
    print("Length of Test = {}".format(eval_dataset.__len__()))
    output_dim = len(label_names)  # 10 (CIFAR-10)
    mean_norms = [0.4914, 0.4822, 0.4465]
    std_norms = [0.247, 0.243, 0.261]
    pTransform = Transform1D(Transform1D.Resize(),
                             Transform1D.Rechannel(channel=3),
                             Transform1D.Decimalize(),
                             Transform1D.Normalization(mean_norms=mean_norms, std_norms=std_norms)
                             )
    return output_dim, pTransform, train_dataset, eval_dataset


def parse_cifar10_tester(root: str):
    import pickle
    # Read the label names
    label_file = os.path.join(root, "batches.meta")
    with open(label_file, 'rb') as file:
        label_data = pickle.load(file)
        label_names = label_data['label_names']
    # Read the data
    test_paths = os.path.join(root, "test_batch")
    with open(test_paths, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    datas = data['data']
    labels = data['label']
    # Transform data array to YoonDataset
    if datas.shape[0] != labels.shape[0]:
        ValueError("The label and data size is not equal")
    dataset = Dataset1D()
    for i in range(datas.shape[0]):
        image = Image.parse_array(32, 32, 3, datas[i], mode="parallel")
        label = labels[i]
        obj = Object(id_=label, name=label_names[label], image=image)
        dataset.append(obj)
    print("Length of Test = {}".format(dataset.__len__()))
    output_dim = label_names.shape[0]  # 10 (CIFAR-10)
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.247, 0.243, 0.261]
    transform = Transform1D(Transform1D.Resize(),
                            Transform1D.Rechannel(channel=3),
                            Transform1D.Decimalize(),
                            Transform1D.Normalization(mean_norms=means, std_norms=stds)
                            )
    return output_dim, transform, dataset


HDF5_FORMAT = ['.h5', '.hdf5', '.mat']


def parse_hdf5_trainer(file_path: str,
                       input_lb: str = 'input',
                       target_lb: str = 'label',
                       train_ratio: float = 0.8):
    assert os.path.splitext(file_path)[1] in HDF5_FORMAT, "Abnormal file format"
    import h5py
    # Read the hierarchical data file
    data = h5py.File(file_path)
    inputs = numpy.array(data[input_lb], dtype=numpy.float32)
    targets = numpy.array(data[target_lb], dtype=numpy.float32)
    # Transform data array to YoonDataset
    if inputs.shape[0] != targets.shape[0]:
        ValueError("The input and target data size is not equal")
    train_dataset = Dataset1D()
    eval_dataset = Dataset1D()
    cut_line = int(inputs.shape[0] * train_ratio)
    for i in range(cut_line):
        image = Image.parse_array(32, 32, 3, datas[i], mode="parallel")
        target = Image.parse_array()
        obj = Object(id_=label, name=label_names[label], image=image)
        train_dataset.append(obj)
    for i in range(cut_line, datas.shape[0]):
        image = Image.parse_array(32, 32, 3, datas[i], mode="parallel")
        label = labels[i]
        obj = Object(id_=label, name=label_names[label], image=image)
        eval_dataset.append(obj)
    print("Length of Train = {}".format(train_dataset.__len__()))
    print("Length of Test = {}".format(eval_dataset.__len__()))