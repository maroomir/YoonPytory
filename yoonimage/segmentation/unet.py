import math
import os.path

import numpy
import torch
import torch.nn
import torch.nn.functional
from torch import tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from yoonimage.data import YoonDataset, YoonTransform
from yoonpytory.log import YoonNLM


class UNetDataset(Dataset):
    def __init__(self,
                 inputs: YoonDataset,
                 targets: YoonDataset = None,
                 output_dim=1
                 ):
        self.transform = YoonTransform(YoonTransform.ResizeToMin(),
                                       YoonTransform.RechannelToMin(),
                                       YoonTransform.ZNormalization())
        self.inputs = self.transform(inputs)
        self.input_dim = self.inputs.min_channel()
        self.targets = targets
        self.output_dim = output_dim

    def __len__(self):
        return self.inputs.__len__()

    def __getitem__(self, item):
        if self.targets is None:
            input_tensor = self.inputs[item].image.get_tensor()
            return input_tensor
        else:
            target_tensor = self.targets[item].image.get_tensor()
            input_tensor = self.inputs[item].image.get_tensor()
            return input_tensor, target_tensor


class ConvolutionBlock(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = 0.3):
        super(ConvolutionBlock, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm2d(output_dim),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm2d(output_dim),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout2d(dropout)
        )

    def forward(self, x: tensor):
        return self.network(x)


class UpSamplerBlock(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        super(UpSamplerBlock, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2, bias=True),
            torch.nn.InstanceNorm2d(output_dim),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: tensor):
        return self.network(x)


class UNet2D(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel: int,
                 depth: int,
                 dropout: float = 0.3):
        super(UNet2D, self).__init__()
        # Init Encoders and Decoders
        self.encoders = torch.nn.ModuleList([ConvolutionBlock(input_dim, channel, dropout)])
        for i in range(depth - 1):
            self.encoders += [ConvolutionBlock(channel, channel * 2, dropout)]
            channel *= 2
        self.down_sampler = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.worker = ConvolutionBlock(channel, channel * 2, dropout)
        self.decoders = torch.nn.ModuleList()
        self.up_samplers = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_samplers += [UpSamplerBlock(channel * 2, channel)]
            self.decoders += [ConvolutionBlock(channel * 2, channel, dropout)]
            channel //= 2
        self.up_samplers += [UpSamplerBlock(channel * 2, channel)]
        self.decoders += [
            torch.nn.Sequential(
                ConvolutionBlock(channel * 2, channel, dropout),
                torch.nn.Conv2d(channel, output_dim, kernel_size=1, stride=1),
                torch.nn.Tanh()
            )
        ]

    def __padding(self, x: tensor):
        def floor_ceil(n):
            return math.floor(n), math.ceil(n)

        batch, density, height, width = x.shape
        width_margin = ((width - 1) | 15) + 1  # 15 = (1111)
        height_margin = ((height - 1) | 15) + 1
        pad_width = floor_ceil((width_margin - width) / 2)
        pad_height = floor_ceil((height_margin - height) / 2)
        x = torch.nn.functional.pad(x, pad_width + pad_height)
        return x, (pad_height, pad_width, height_margin, width_margin)

    def __unpadding(self, x, pad_height, pad_width, height_margin, width_margin):
        return x[...,
               pad_height[0]:height_margin - pad_height[1],
               pad_width[0]:width_margin - pad_width[1]]

    def forward(self, x: tensor):
        x, pad_option = self.__padding(x)
        layers = []
        result = x
        # Apply down sampling layers
        for i, pEncoder in enumerate(self.encoders):
            result = pEncoder(result)
            layers.append(result)
            result = self.down_sampler(result)
        result = self.worker(result)
        # Apply up sampling layers
        for sampler, decoder in zip(self.up_samplers, self.decoders):
            attach_layer = layers.pop()
            result = sampler(result)
            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]  # left, right, top, bottom
            if result.shape[-1] != attach_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if result.shape[-2] != attach_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                result = torch.nn.functional.pad(result, padding, "reflect")
            result = torch.cat([result, attach_layer], dim=1)
            result = decoder(result)
        layers.clear()  # To Memory Optimizing
        result = self.__unpadding(result, *pad_option)
        return result


# Define a train function
def __process_train(model: UNet2D, data_loader: DataLoader, criterion: BCEWithLogitsLoss,
                    optimizer: Adam, logger: YoonNLM):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Perform a training using the defined network
    model.train()
    # Warp the iterable Data Loader with TQDM
    bar = tqdm(enumerate(data_loader))
    sample_length = 0
    total_loss = 0.0
    total_acc = 0.0
    for i, (input, target) in bar:
        # Move data and label to device
        input = input.type(torch.FloatTensor).to(device)
        target = target.type(torch.FloatTensor).to(device)
        # Pass the input data through the defined network architecture
        output = model(input)
        # Compute a loss function
        loss = criterion(output, target)
        total_loss += loss.item() * len(target[0])  # Loss per batch * batch
        # Compute network accuracy
        acc = torch.sum(torch.eq(output > 0.5, target > 0.5)).item()  # output and targets binary
        sample_length += len(input[0])
        total_acc += acc
        # Perform backpropagation to update network parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        message = logger.write(i, len(data_loader),
                               CELoss=total_loss / sample_length, ACC=(total_acc / sample_length) * 100.0)
        bar.set_description(message)


# Define a test function
def __process_evaluate(model: UNet2D, data_loader: DataLoader, criterion: BCEWithLogitsLoss, logger: YoonNLM):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Perform an evaluation using the defined network
    model.eval()
    # Warp the iterable Data Loader with TQDM
    bar = tqdm(enumerate(data_loader))
    sample_length = 0
    total_loss = 0
    total_acc = 0
    for i, (input, target) in bar:
        # Move data and label to device
        input = input.type(torch.FloatTensor).to(device)
        target = input.type(torch.FloatTensor).to(device)
        # Pass the input data through the defined network architecture
        output = model(input)
        # Compute a loss function
        loss = criterion(output, target)
        total_loss += loss.item() * len(target[0])  # Loss per batch * batch
        # Compute network accuracy
        acc = torch.sum(torch.eq(output > 0.5, target > 0.5)).item()  # output and targets binary
        sample_length += len(input[0])
        total_acc += acc
        # Trace the log
        message = logger.write(i, len(data_loader),
                               CELoss=total_loss / sample_length, ACC=(total_acc / sample_length) * 100.0)
        bar.set_description(message)
    return total_loss / sample_length


def train(epoch: int,
          model_path: str,
          train_data: YoonDataset,
          train_label: YoonDataset,
          eval_data: YoonDataset,
          eval_label: YoonDataset,
          channel=8,
          depth=4,
          batch_size=1,
          num_workers=0,  # 0: CPU / 4 : GPU
          dropout=0.3,
          decay=0.5,
          is_init_epoch=False):
    def learning_func(iStep):
        return 1.0 - max(0, iStep - epoch * (1 - decay)) / (decay * epoch + 1)

    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define the training and testing data-set
    train_set = UNetDataset(train_data, train_label)
    pTrainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_set = UNetDataset(eval_data, eval_label)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # Define a network model
    model = UNet2D(input_dim=train_set.input_dim, output_dim=train_set.output_dim, channel=channel,
                   depth=depth, dropout=dropout).to(device)
    # Set the optimizer with adam
    optimizer = torch.optim.Adam(model.parameters())
    # Set the training criterion
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    # Set the scheduler to control the learning rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_func)
    # Load pre-trained model
    start = 0
    print("Directory of the pre-trained model: {}".format(model_path))
    if model_path is not None and os.path.exists(model_path) and is_init_epoch is False:
        pModelData = torch.load(model_path, map_location=device)
        start = pModelData['epoch']
        model.load_state_dict(pModelData['model'])
        optimizer.load_state_dict(pModelData['optimizer'])
        print("## Successfully load the model at {} epochs!".format(start))
    # Define the log manager
    train_logger = YoonNLM(start, root="./NLM/UNet2D", mode="Train")
    eval_logger = YoonNLM(start, root="./NLM/UNet2D", mode="Eval")
    # Train and Test Repeat
    min_loss = 10000.0
    for i in range(start, epoch + 1):
        # Train the network
        __process_train(model=model, data_loader=pTrainLoader, criterion=criterion,
                        optimizer=optimizer, logger=train_logger)
        # Test the network
        loss = __process_evaluate(model=model, data_loader=valid_loader, criterion=criterion, logger=eval_logger)
        # Change the learning rate
        scheduler.step()
        # Rollback the model when loss is NaN
        if math.isnan(loss):
            if model_path is not None and os.path.exists(model_path):
                # Reload the best model and decrease the learning rate
                pModelData = torch.load(model_path, map_location=device)
                model.load_state_dict(pModelData['model'])
                pOptimizerData = pModelData['optimizer']
                pOptimizerData['param_groups'][0]['lr'] /= 2  # Decrease the learning rate by 2
                optimizer.load_state_dict(pOptimizerData)
                print("## Rollback the Model with half learning rate!")
        # Save the optimal model
        elif loss < min_loss:
            min_loss = loss
            torch.save({'epoch': i, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       model_path)
        elif i % 100 == 0:
            torch.save({'epoch': i, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       'unet_{}epoch.pth'.format(i))


def test(test_data: YoonDataset,
         model_path: str,
         channel=8,
         depth=4,
         num_workers=0,  # 0: CPU / 4 : GPU
         dropout=0.3):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define a data path for plot for test
    dataset = UNetDataset(test_data)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    # Load UNET model
    model = UNet2D(input_dim=dataset.input_dim, output_dim=dataset.output_dim, channel=channel,
                    depth=depth, dropout=dropout).to(pDevice)
    model.eval()
    file = torch.load(model_path)
    model.load_state_dict(file['model'])
    print("Successfully load the Model in path")
    # Start the test sequence
    bar = tqdm(data_loader)
    print("Length of data = ", len(bar))
    output_list = []
    for i, input in enumerate(bar):
        input = input.type(torch.FloatTensor).to(pDevice)
        output = model(input)
        output_list.append(output.detach().cpu().numpy())
    # Warp the tensor to Dataset
    return YoonDataset.from_tensor(images=numpy.concatenate(output_list))
