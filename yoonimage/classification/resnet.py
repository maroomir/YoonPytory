import math
import os

import matplotlib.pyplot
import numpy.random
import sklearn.metrics
import torch
import torch.nn.functional
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from yoonimage.classification.dataset import ClassificationDataset, collate_segmentation
from yoonimage.data import Dataset1D, Transform1D
from yoonpytory.log import NLM


class ConvolutionBlock(Module):
    def __init__(self,
                 input_dim: int,
                 filters: list,
                 stride: int):
        super(ConvolutionBlock, self).__init__()
        self.filter1, self.filter2, self.filter3 = filters
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, self.filter1, kernel_size=1, stride=stride, padding=0, bias=False),
            torch.nn.BatchNorm2d(self.filter1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.filter1, self.filter2, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.filter2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.filter2, self.filter3, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(self.filter3)
        )
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, self.filter3, kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(self.filter3)
        )

    def forward(self, x: Tensor):
        output = self.network(x)
        output += self.shortcut(x)
        output = torch.nn.functional.relu(output)
        return output


class IdentityBlock(Module):
    def __init__(self,
                 filters: list):
        super(IdentityBlock, self).__init__()
        self.filter1, self.filter2, self.filter3 = filters
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(self.filter3, self.filter1, kernel_size=1, stride=1, padding=0, bias=False),  # Pad=0=valid
            torch.nn.BatchNorm2d(self.filter1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.filter1, self.filter2, kernel_size=3, stride=1, padding=1, bias=False),  # Pad=1=same
            torch.nn.BatchNorm2d(self.filter2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.filter2, self.filter3, kernel_size=1, stride=1, padding=0, bias=False),  # Pad=0=valid
            torch.nn.BatchNorm2d(self.filter3)
        )

    def forward(self, x: Tensor):
        output = self.network(x)
        output += x
        output = torch.nn.functional.relu(output)
        return output


class ResNet50(Module):  # Conv Count = 50
    def __init__(self,
                 input_dim: int,
                 num_class: int):
        super(ResNet50, self).__init__()
        self.layer1 = torch.nn.Sequential(  # Conv=1
            torch.nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.layer2 = torch.nn.Sequential(  # Conv=10
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvolutionBlock(input_dim=64, filters=[64, 64, 256], stride=1),  # Conv=4
            IdentityBlock(filters=[64, 64, 256]),  # Conv=3
            IdentityBlock(filters=[64, 64, 256])  # Conv=3
        )
        self.layer3 = torch.nn.Sequential(  # Conv=13
            ConvolutionBlock(input_dim=256, filters=[128, 128, 512], stride=2),
            IdentityBlock(filters=[128, 128, 512]),
            IdentityBlock(filters=[128, 128, 512]),
            IdentityBlock(filters=[128, 128, 512])
        )
        self.layer4 = torch.nn.Sequential(  # Conv=16
            ConvolutionBlock(input_dim=512, filters=[256, 256, 1024], stride=2),
            IdentityBlock(filters=[256, 256, 1024]),
            IdentityBlock(filters=[256, 256, 1024]),
            IdentityBlock(filters=[256, 256, 1024]),
            IdentityBlock(filters=[256, 256, 1024])
        )
        self.layer5 = torch.nn.Sequential(  # Conv=10
            ConvolutionBlock(input_dim=1024, filters=[512, 512, 2048], stride=2),
            IdentityBlock(filters=[512, 512, 2048]),
            IdentityBlock(filters=[512, 512, 2048])
        )
        self.fc_layer = torch.nn.Linear(2048, num_class)

    def forward(self, x: Tensor):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = torch.nn.functional.avg_pool2d(output, kernel_size=1)
        output = output.view(output.size(0), -1)
        output = self.fc_layer(output)
        return output


def __process_train(model: ResNet50, data_loader: DataLoader, optimizer, criterion, logger: NLM):
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


def __process_evaluate(model: ResNet50,
                       data_loader: DataLoader,
                       criterion,
                       logger: NLM):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.eval()
    bar = tqdm(enumerate(data_loader))
    total_loss = 0.0
    total_correct = 0
    sample_length = 0
    with torch.no_grad():
        for i, (input, target) in bar:
            input = input.type(torch.FloatTensor).to(device)
            target = target.type(torch.LongTensor).to(device)
            output = model(input)
            loss = criterion(output, target)
            total_loss += loss.item() * target.size(0)
            _, target_hat = output.max(1)
            sample_length += target.size(0)
            total_correct += target_hat.eq(target).sum().item()
            message = logger.write(i, len(data_loader),
                                   Loss=total_loss / sample_length, Acc=100 * total_correct / sample_length)
            bar.set_description(message)
    return total_loss / sample_length


def __process_test(model: ResNet50, data_loader: DataLoader, labels: list):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.eval()
    target_labels = []
    predicted_labels = []
    for i, (input, target) in enumerate(data_loader):
        input = input.type(torch.FloatTensor).to(device)
        target = target.type(torch.LongTensor).to(device)
        output = model(input)
        _, target_hat = output.max(1)
        target_labels = numpy.concatenate((target_labels, target.cpu().numpy()))
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
            matplotlib.pyplot.text(j, i, format(matrix[i, j], str_format), horizontalalignment="center", color="white"
            if matrix[i, j] > thresh else "black")
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
        image, nLabel = dataset[rands[i]]
        plot.set_title("%s" % labels[nLabel])
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
    trainset = ClassificationDataset(train_data, num_class, transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_segmentation, num_workers=num_workers, pin_memory=True)
    valid_set = ClassificationDataset(eval_data, num_class, transform)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_segmentation, num_workers=num_workers, pin_memory=True)
    # Define a network model
    model = ResNet50(input_dim=trainset.input_dim, num_class=trainset.output_dim).to(device)
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
    train_logger = NLM(start, root="./NLM/ResNet", mode="Train")
    valid_logger = NLM(start, root="./NLM/ResNet", mode="Eval")
    # Train and Test Repeat
    min_loss = 10000.0
    for i in range(start, epoch + 1):
        __process_train(model=model, data_loader=train_loader, criterion=criterion,
                        optimizer=optimizer, logger=train_logger)
        loss = __process_evaluate(model=model, data_loader=valid_loader, criterion=criterion, logger=valid_logger)
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
                       'resnet_{}epoch.pth'.format(i))


def test(test_data: Dataset1D,
         model_path: str,
         num_class: int,
         transform: Transform1D,
         num_workers=0,  # 0: CPU / 4 : GPU
         ):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define a data path for plot for test
    dataset = ClassificationDataset(test_data, num_class, transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_segmentation, num_workers=num_workers, pin_memory=True)
    # Load the model
    model = ResNet50(input_dim=dataset.input_dim, num_class=dataset.output_dim).to(device)
    model.eval()
    file = torch.load(model_path)
    model.load_state_dict(file['model'])
    print("Successfully load the Model in path")
    # Start the test sequence
    bar = tqdm(data_loader)
    print("Length of data = ", len(bar))
    labels = []
    for i, input in enumerate(bar):
        input = input.type(torch.FloatTensor).to(device)
        output = model(input)
        _, target_hat = output.max(1)
        labels.append(target_hat.detach().cpu().numpy())
    # Warp the tensor to Dataset
    return Dataset1D.from_tensor(images=None, labels=numpy.concatenate(labels))
