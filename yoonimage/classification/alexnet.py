import math
import os

import matplotlib.pyplot
import numpy.random
import sklearn.metrics
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from yoonimage.classification.dataset import ClassificationDataset, collate_segmentation
from yoonimage.dataset import Dataset1D, Transform1D
from yoonpytory.log import NLM


class AlexNet(Module):
    def __init__(self,
                 input_dim,
                 num_class):
        super(AlexNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, 96, kernel_size=11, stride=4, padding=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(256 * 2 * 2, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_class)
        )

    def forward(self, x: Tensor):
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


def __process_train(model: AlexNet,
                    data_loader: DataLoader,
                    optimizer,
                    criterion,
                    logger: NLM):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.train()
    bar = tqdm(enumerate(data_loader))
    total_loss = 0.0
    total_correct = 0
    sample_length = 0
    for i, (_input, _target) in bar:
        _input = _input.type(torch.FloatTensor).to(device)
        _target = _target.type(torch.LongTensor).to(device)
        output = model(_input)
        optimizer.zero_grad()
        loss = criterion(output, _target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * _target.size(0)
        _, target_hat = output.max(1)
        sample_length += _target.size(0)
        total_correct += target_hat.eq(_target).sum().item()
        message = logger.write(i, len(data_loader),
                               Loss=total_loss / sample_length, Acc=100 * total_correct / sample_length)
        bar.set_description(message)


def __process_evaluate(model: AlexNet,
                       data_loader: DataLoader,
                       criterion,
                       logger: NLM):
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    model.eval()
    bar = tqdm(enumerate(data_loader))
    total_loss = 0.0
    total_correct = 0
    sample_length = 0
    with torch.no_grad():
        for i, (_input, _target) in bar:
            _input = _input.type(torch.FloatTensor).to(pDevice)
            _target = _target.type(torch.LongTensor).to(pDevice)
            output = model(_input)
            loss = criterion(output, _target)
            total_loss += loss.item() * _target.size(0)
            _, target_hat = output.max(1)
            sample_length += _target.size(0)
            total_correct += target_hat.eq(_target).sum().item()
            message = logger.write(i, len(data_loader),
                                   Loss=total_loss / sample_length, Acc=100 * total_correct / sample_length)
            bar.set_description(message)
    return total_loss / sample_length


def __process_test(model: AlexNet, data_loader: DataLoader, labels: list):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.eval()
    target_labels = []
    predicted_labels = []
    for i, (_input, _target) in enumerate(data_loader):
        _input = _input.type(torch.FloatTensor).to(device)
        _target = _target.type(torch.LongTensor).to(device)
        output = model(_input)
        _, target_hat = output.max(1)
        target_labels = numpy.concatenate((target_labels, _target.cpu().numpy()))
        predicted_labels = numpy.concatenate((predicted_labels, target_hat.cpu().numpy()))
    # Compute confusion matrix
    matrix = sklearn.metrics.confusion_matrix(target_labels, predicted_labels)
    numpy.set_printoptions(precision=2)
    corrected = (predicted_labels == target_labels)
    acc = numpy.sum(corrected * 1) / len(corrected)
    print("Accuracy: %.5f" % acc)
    # Plot non-normalized confusion matrix
    matplotlib.pyplot.figure()
    __draw_confusion_matrix(matrix=matrix, labels=labels, title="Confusion matrix, without normalization")
    # Plot non-normalized confusion matrix
    matplotlib.pyplot.figure()
    __draw_confusion_matrix(matrix=matrix, labels=labels, is_norm=True,
                            title="Confusion matrix, without normalization")
    matplotlib.pyplot.show()


def __draw_confusion_matrix(matrix: numpy,
                            labels: list,
                            is_norm=False,
                            color_map=matplotlib.pyplot.cm.Blues,
                            title="Confusion Matrix"):
    if is_norm:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    matplotlib.pyplot.imshow(matrix, interpolation='nearest', cmap=color_map)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.colorbar()
    marks = numpy.arange(len(labels))
    matplotlib.pyplot.xticks(marks, labels, rotation=45)
    matplotlib.pyplot.yticks(marks, labels)
    str_format = '.2f' if is_norm else 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matplotlib.pyplot.text(j, i, format(matrix[i, j], str_format), horizontalalignment="center",
                                   color="white" if matrix[i, j] > thresh else "black")
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.ylabel("True label")
    matplotlib.pyplot.xlabel("Predicted label")


def __draw_dataset(dataset: Dataset, labels: list, show_count=15):
    data_count = len(dataset)
    figure = matplotlib.pyplot.figure()
    rands = numpy.random.randint(data_count, size=show_count)
    col_count = int(show_count / 2)
    row_count = show_count - col_count
    for i in range(show_count):
        plot = figure.add_subplot(col_count, row_count, i + 1)
        plot.set_xticks([])
        plot.set_yticks([])
        image, label = dataset[rands[i]]
        plot.set_title("%s" % labels[label])
        plot.imshow(image)
    matplotlib.pyplot.show()


def train(epoch: int,
          model_path: str,
          num_class: int,
          train_data: Dataset1D,
          eval_data: Dataset1D,
          transform: Transform1D,
          batch_size=32,
          num_workers=0,  # 0: CPU / 4 : GPU
          learning_rate=0.1,
          is_init_epoch=False):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define the training and testing data-set
    train_set = ClassificationDataset(train_data, num_class, transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_segmentation, num_workers=num_workers, pin_memory=True)
    valid_set = ClassificationDataset(eval_data, num_class, transform)
    pValidationLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_segmentation, num_workers=num_workers, pin_memory=True)
    # Define a network model
    model = AlexNet(input_dim=train_set.input_dim, num_class=train_set.output_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # Load pre-trained model
    start = 0
    print("Directory of the pre-trained model: {}".format(model_path))
    if model_path is not None and os.path.exists(model_path) and is_init_epoch is False:
        model_data = torch.load(model_path, map_location=device)
        start = model_data['epoch']
        model.load_state_dict(model_data['model'])
        optimizer.load_state_dict(model_data['optimizer'])
        print("## Successfully load the model at {} epochs!".format(start))
    # Define the log manager
    train_logger = NLM(start, root="./NLM/AlexNet", mode="Train")
    eval_logger = NLM(start, root="./NLM/AlexNet", mode="Eval")
    # Train and Test Repeat
    min_loss = 10000.0
    for i in range(start, epoch + 1):
        __process_train(model=model, data_loader=train_loader, criterion=criterion,
                        optimizer=optimizer, logger=train_logger)
        loss = __process_evaluate(model=model, data_loader=pValidationLoader, criterion=criterion,
                                  logger=eval_logger)
        # Change the learning rate
        scheduler.step()
        # Rollback the model when loss is NaN
        if math.isnan(loss):
            if model_path is not None and os.path.exists(model_path):
                # Reload the best model and decrease the learning rate
                model_data = torch.load(model_path, map_location=device)
                model.load_state_dict(model_data['model'])
                optimizer_data = model_data['optimizer']
                optimizer_data['param_groups'][0]['lr'] /= 2  # Decrease the learning rate by 2
                optimizer.load_state_dict(optimizer_data)
                print("## Rollback the Model with half learning rate!")
        # Save the optimal model
        elif loss < min_loss:
            min_loss = loss
            torch.save({'epoch': i, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       model_path)
        elif i % 100 == 0:
            torch.save({'epoch': i, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       'alexnet_{}epoch.pth'.format(i))


def test(test_data: Dataset1D,
         model_path: str,
         num_class: int,
         transform: Transform1D,
         num_worker=0,  # 0: CPU / 4 : GPU
         ):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define a data path for plot for test
    dataset = ClassificationDataset(test_data, num_class, transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_segmentation,
                             num_workers=num_worker, pin_memory=True)
    # Load the model
    model = AlexNet(input_dim=dataset.input_dim, num_class=dataset.output_dim).to(device)
    model.eval()
    file = torch.load(model_path)
    model.load_state_dict(file['model'])
    print("Successfully load the Model in path")
    # Start the test sequence
    bar = tqdm(data_loader)
    print("Length of data = ", len(bar))
    labels = []
    for i, _input in enumerate(bar):
        _input = _input.type(torch.FloatTensor).to(device)
        output = model(_input)
        _, target_hat = output.max(1)
        labels.append(target_hat.detach().cpu().numpy())
    # Warp the tensor to Dataset
    return Dataset1D.from_tensor(images=None, labels=numpy.concatenate(labels))
