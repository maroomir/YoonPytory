import os
import numpy

from yoonimage.image import Image


def parse_root(root: str):
    # Parse the file list
    res = []
    for root_, dir_, path in os.walk(root):
        if len(path) > 0:
            for pth_ in path:
                if os.path.splitext(pth_)[1] in [".jpg", ".bmp", ".png"]:
                    image = Image()
                    image.path = os.path.join(root_, pth_)
                    res.append(image.buffer)
    return res


def parse_cifar10_trainer(root: str, ratio: float = 0.8):
    import pickle
    # Read the label names
    label_path = os.path.join(root, "batches.meta")
    with open(label_path, 'rb') as file:
        label_data = pickle.load(file)
        label_names = label_data['label_names']
    # Read the data
    path_ = [os.path.join(root, "data_batch_{}".format(i + 1)) for i in range(5)]
    datas = []
    labels = []
    for pth in path_:
        with open(pth, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            datas.append(data[b'data'])
            labels.append(data[b'labels'])
    datas = numpy.concatenate(datas, axis=0)
    labels = numpy.concatenate(labels, axis=0)
    if datas.shape[0] != labels.shape[0]:
        ValueError("The label and data size is not equal")
    # make the dataset
    cut_line = int(datas.shape[0] * ratio)
    train_set, eval_set = [], []
    for i in range(cut_line):
        train_set += [{'image': datas[i], 'label': labels[i]}]
    for i in range(cut_line, datas.shape[0]):
        eval_set += [{'image': datas[i], 'label': labels[i]}]
    print("Length of Train = {}".format(len(train_set)))
    print("Length of Test = {}".format(len(eval_set)))
    # construct the dataset params
    output_dim = len(label_names)  # 10 (CIFAR-10)
    mean_norms, std_norms = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    return {'train': train_set,
            'eval': eval_set,
            'num_class': output_dim,
            'param': {
                'mean_norms': mean_norms,
                'std_norms': std_norms
            }}


def parse_cifar10_tester(root: str):
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
    # construct the dataset params
    output_dim = label_names.shape[0]  # 10 (CIFAR-10)
    mean_norms, std_norms = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    return {'test': test_set,
            'num_class': output_dim,
            'param': {
                'mean_norms': mean_norms,
                'std_norms': std_norms
            }}


HDF5_FORMAT = ['.h5', '.hdf5', '.mat']
def parse_hdf5_trainer(file_path: str,
                       input_lb: str = 'input',
                       target_lb: str = 'label',
                       ratio: float = 0.8):
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
