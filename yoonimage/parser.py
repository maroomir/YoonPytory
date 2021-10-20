import os
import pickle
import numpy

from yoonimage.data import YoonDataset, YoonObject, YoonTransform
from yoonimage.image import YoonImage


def parse_root(root: str):
    # Parse the file list
    path_list = os.listdir(root)
    dataset = YoonDataset()
    count = 0
    for path in path_list:
        if "jpg" in path or "bmp" in path or "png" in path:
            image_path = os.path.join(root, path)
            image = YoonImage.from_path(image_path)
            object = YoonObject(id=count, name=path, image=image)
            dataset.append(object)
            count += 1
    return count, dataset


def parse_cifar10_trainer(root: str,
                          train_ratio: float = 0.8,
                          ):
    # Read the label names
    label_path = os.path.join(root, "batches.meta")
    with open(label_path, 'rb') as pFile:
        label_data = pickle.load(pFile)
        label_names = label_data['label_names']
    # Read the data
    train_paths = [os.path.join(root, "data_batch_{}".format(i + 1)) for i in range(5)]
    data_list = []
    label_list = []
    for path in train_paths:
        with open(path, 'rb') as pFile:
            data = pickle.load(pFile, encoding='bytes')
            data_list.append(data[b'data'])
            label_list.append(data[b'labels'])
    data_list = numpy.concatenate(data_list, axis=0)
    label_list = numpy.concatenate(label_list, axis=0)
    # Transform data array to YoonDataset
    if data_list.shape[0] != label_list.shape[0]:
        ValueError("The label and data size is not equal")
    train_dataset = YoonDataset()
    eval_dataset = YoonDataset()
    cutline = int(data_list.shape[0] * train_ratio)
    for i in range(cutline):
        image = YoonImage.parse_array(32, 32, 3, data_list[i], mode="parallel")
        label = label_list[i]
        object = YoonObject(id=label, name=label_names[label], image=image)
        train_dataset.append(object)
    for i in range(cutline, data_list.shape[0]):
        image = YoonImage.parse_array(32, 32, 3, data_list[i], mode="parallel")
        label = label_list[i]
        object = YoonObject(id=label, name=label_names[label], image=image)
        eval_dataset.append(object)
    print("Length of Train = {}".format(train_dataset.__len__()))
    print("Length of Test = {}".format(eval_dataset.__len__()))
    output_dim = len(label_names)  # 10 (CIFAR-10)
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.247, 0.243, 0.261]
    pTransform = YoonTransform(YoonTransform.Resize(),
                               YoonTransform.Rechannel(channel=3),
                               YoonTransform.Decimalize(),
                               YoonTransform.Normalization(mean_norms=means, std_norms=stds)
                               )
    return output_dim, pTransform, train_dataset, eval_dataset


def parse_cifar10_tester(root: str):
    # Read the label names
    label_file = os.path.join(root, "batches.meta")
    with open(label_file, 'rb') as pFile:
        label_data = pickle.load(pFile)
        label_names = label_data['label_names']
    # Read the data
    test_paths = os.path.join(root, "test_batch")
    with open(test_paths, 'rb') as pFile:
        data = pickle.load(pFile, encoding='bytes')
    data_list = data['data']
    label_list = data['label']
    # Transform data array to YoonDataset
    if data_list.shape[0] != label_list.shape[0]:
        ValueError("The label and data size is not equal")
    dataset = YoonDataset()
    for i in range(data_list.shape[0]):
        image = YoonImage.parse_array(32, 32, 3, data_list[i], mode="parallel")
        label = label_list[i]
        object = YoonObject(id=label, name=label_names[label], image=image)
        dataset.append(object)
    print("Length of Test = {}".format(dataset.__len__()))
    output_dim = label_names.shape[0]  # 10 (CIFAR-10)
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.247, 0.243, 0.261]
    transform = YoonTransform(YoonTransform.Resize(),
                              YoonTransform.Rechannel(channel=3),
                              YoonTransform.Decimalize(),
                              YoonTransform.Normalization(mean_norms=means, std_norms=stds)
                              )
    return output_dim, transform, dataset
