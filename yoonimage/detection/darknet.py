from __future__ import division
from typing import Any

import torch
import torch.nn
import torch.nn.functional
from torch import tensor
from torch.nn import Module
from torch.autograd import Function
from torch.utils.data import Dataset

import numpy

from yoonimage.data import Dataset1D


def parse_config(config: str):
    file = open(config, 'r')
    lines = file.read().split("\n")
    lines = [line for line in lines if len(line) > 0]  # erase the empty line
    lines = [line for line in lines if line[0] != "#"]  # erase the comment
    lines = [iLine.rstrip().lstrip for iLine in lines]  # erase the space
    # Get parameter blocks
    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":  # start new block
            if line(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


class YoloDataset(Dataset):
    def __init__(self,
                 dataset: Dataset1D,
                 config: str):
        self.data = dataset
        self.blocks = parse_config(config)
        self.width = 0
        self.height = 0
        self.channel = 0

    def __extract_info(self):
        net_param = self.blocks[0]
        self.height = int(net_param['height'])
        self.width = int(net_param['width'])
        self.channel = int(net_param['channels'])


class MishActivation(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        ctx.save_for_backward(args[0])
        # Activation :  X * tanh(ln(1 + exp(X)))
        y = args[0].mul(torch.tanh(torch.nn.functional.softplus(args[0])))
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x = ctx.saved_tensor[0]
        sigmoid = torch.sigmoid(x)
        tanh = torch.nn.functional.softplus(x).tanh()
        return grad_outputs[0].mul(tanh + x * sigmoid * (1 - tanh * tanh))


class Mish(Module):
    def __init__(self,
                 inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x: tensor):
        return MishActivation.apply(x)


class MaxPoolStride(Module):
    def __init__(self, kernel):
        super(MaxPoolStride, self).__init__()
        self.kernel_size = kernel
        self.pad = kernel - 1

    def forward(self, x: tensor):
        x = torch.nn.functional.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        return torch.nn.functional.max_pool2d(x, self.kernel_size, self.pad)


class DummyLayer(Module):
    def __init__(self):
        super(DummyLayer, self).__init__()


class YoloLayer(Module):
    def __init__(self,
                 anchors: list,
                 image_width: int = 0,
                 image_height: int = 0,
                 num_class: int = 0):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.class_count = num_class
        self.anchor_boxes = anchors

    def forward(self, x: tensor):  # Tensor Shape = (Batch, CH, Height, Width)
        batch_size = x.size(0)
        num_grid = self.image_height // x.size(2)
        grid_size = self.image_height // num_grid
        num_attr = 5 + self.class_count  # Count : (x, y, w, h, confidence, class...)
        num_anchor = len(self.anchor_boxes)
        x = x.view(batch_size, num_attr * num_anchor, grid_size * grid_size)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, grid_size * grid_size * num_anchor, num_attr)
        anchors = [(width / num_grid, height / num_grid) for width, height in self.anchor_boxes]
        # Sigmoid the CenterX, CenterY and Object Confidence
        x[:, :, 0] = torch.sigmoid(x[:, :, 0])  # X
        x[:, :, 1] = torch.sigmoid(x[:, :, 1])  # Y
        x[:, :, 4] = torch.sigmoid(x[:, :, 4])  # Confidence
        # Add the center offsets
        grids = numpy.arance(grid_size)
        x_grids, y_grids = numpy.meshgrid(grids, grids)
        device = x.device
        grid_offset_x = x_grids.view(-1, 1).to(device)
        grid_offset_y = y_grids.view(-1, 1).to(device)
        grid_offset = torch.cat((grid_offset_x, grid_offset_y), 1).repeat(1, num_anchor).view(-1, 2).unsqueeze(0)
        x[:, :, :2] += grid_offset  # X, Y (offset into the item 0, 1)
        # The log area transform to the height and width
        anchor = torch.Tensor(anchors)
        anchor = anchor.repeat(grid_size * grid_size, 1).unsqueeze(0)
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * anchor  # Width, Height (item 2, 3)
        # Softmax the class scores
        x[:, :, , 5:5 + self.class_count] = torch.sigmoid((x[:, :, 5:5 + self.class_count]))
        x[:, :, :4] *= num_grid  # Adjust the stride at the X, Y, Width, Height (item 0 ~ 3)
        return x


class DarkNet(Module):
    def __init__(self,
                 config: str):
        super(DarkNet, self).__init__()
        self.blocks = parse_config(config)
        self.net_info, self.modules = self.__create_module()
        self.header = torch.Tensor([0, 0, 0, 0])
        self.seen = 0

    def __create_module(self):
        net = self.blocks[0]
        modules = torch.nn.ModuleList()
        dim_prev = self.net_info['channels']
        filters = []
        for i, block in enumerate(self.blocks[1:]):
            module = torch.nn.Sequential()
            if block['type'] == "convolutional":
                active_func = block['activation']
                try:
                    batch_norm = int(block['batch_normalize'])
                    bias = False
                except:
                    batch_norm = 0
                    bias = True
                dim_module = int(block['filters'])
                padding = int(block['pad'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                if padding > 0:  # Auto padding
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0
                # Add the convolutional layer
                module.add_module("conv_{0}".format(i),
                                  torch.nn.Conv2d(in_channels=dim_prev, out_channels=dim_module,
                                                  kernel_size=kernel_size, stride=stride,
                                                  padding=pad, bias=bias))
                # Add the batch norm layer
                if batch_norm > 0:
                    module.add_module("batch_norm_{0}".format(i), torch.nn.BatchNorm2d(num_features=dim_module))
                # Check the activation
                if active_func == "leaky":
                    module.add_module("leaky_{0}".format(i), torch.nn.LeakyReLU(0.1, inplace=True))
                elif active_func == "mish":
                    module.add_module("mish_{0}".format(i), Mish(inplace=True))
                elif active_func == "logistic":
                    module.add_module("mish_{0}".format(i), torch.nn.Sigmoid())
            elif block['type'] == "upsample":
                stride = int(block['stride'])
                module.add_module("leaky_{0}".format(i), torch.nn.Upsample(scale_factor=stride, mode="nearest"))
            elif block['type'] == "route":
                block['layers'] = block['layers'].split(',')
                start = int(block['layers'][0])
                try:
                    end = int(block['layers'][1])
                except:
                    end = 0
                if start > 0:
                    start = start - i
                if end > 0:
                    end = end - i
                module.add_module("route_{0}".format(i), DummyLayer())
                if end < 0:
                    dim_module = filters[i + start] + filters[i + end]
                else:
                    dim_module = filters[i + start]
            # Define the skip connection layer
            elif block['type'] == "shortcut":
                module.add_module("shortcut_{0}".format(i), DummyLayer())
            # Define the pooling layer
            elif block['type'] == "maxpool":
                stride = int(block['stride'])
                kernel_size = int(block['size'])
                if stride != 1:
                    module.add_module("maxpool_{0}".format(i),
                                      torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
                else:
                    module.add_module("maxpool_{0}".format(i), MaxPoolStride(kernel=kernel_size))
            # Define the detection layer
            elif block['type'] == "yolo":
                masks = block['mask'].split(",")
                masks = [int(mask) for mask in masks]
                anchor_boxes = block['anchors'].split(",")
                anchor_boxes = [int(anchor) for anchor in anchor_boxes]
                anchor_boxes = [(anchor_boxes[box], anchor_boxes[box + 1])
                                for box in range(0, len(anchor_boxes), 2)]
                anchor_boxes = [anchor_boxes[mask] for mask in masks]
                module.add_module("detection_{0}".format(i), YoloLayer(anchor_boxes))
            modules.append(module)
            dim_prev = dim_module
            filters.append(dim_module)
        return net, modules

    def forward(self, x: tensor):
        result: tensor = None
        block = self.blocks[1:]
        result_stack = {}
        for i, block in enumerate(block):
            type = block['type']
            if type == "convolutional" or type == "upsample":
                x = self.modules[i](x)
                result_stack[i] = x
            elif type == "route":
                layers = block['layers']
                layers = [int(j) for j in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = result_stack[i + layers[0]]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = result_stack[i + layers[0]]
                    map2 = result_stack[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                    result_stack[i] = x
            elif type == "shortcut":
                from_back = int(block['from'])
                x = result_stack[i - 1] + result_stack[i + from_back]
                result_stack[i] = x
            elif type == "yolo":
                self.modules[i][0].image_height = int(self.net_info['height'])  # YoloLayer
                self.modules[i][0].image_width = int(self.net_info['width'])
                self.modules[i][0].class_count = int(block['classes'])
                # Predict the bounding boxes
                x = x.data
                x = self.modules[i][0](x)
                if type(x) == int:
                    continue
                if result is None:
                    result = x
                else:
                    result = torch.cat((result, x), dim=1)
                result_stack[i] = result_stack[i - 1]
        return result if result is not None else 0

    def load_weight(self, weight_file: str):
        # Open the weights file
        file = open(weight_file, "rb")
        # The first values are a header information
        # Major version / Minor version / Subversion / Images
        header = numpy.fromfile(file, dtype=numpy.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        # The rest of the values are the weights
        weights = numpy.fromfile(file, dtype=numpy.float32)
        blocks = self.blocks[1:]
        start = 0
        for i, block in enumerate(blocks):
            type = block['type']
            if type == "convolutional":
                try:
                    batch_norm = int(block['batch_normalize'])
                except:
                    batch_norm = 0
                convolution = self.modules[i][0]
                if batch_norm > 0:
                    normalized = self.modules[i][1]
                    # Get the number of weights of Batch Norm Layer
                    num_bias = normalized.bias.numel()
                    # Load the weights
                    biases = torch.from_numpy(weights[start: start + num_bias])
                    start += num_bias
                    weight = torch.from_numpy(weights[start: start + num_bias])
                    running_mean = torch.from_numpy(weights[start: start + num_bias])
                    start += num_bias
                    running_var = torch.from_numpy(weights[start: start + num_bias])
                    start += num_bias
                    # Cast the loaded weights into the dimension of the model weight
                    biases = biases.view_as(normalized.bias.data)
                    weight = weight.view_as(normalized.weight.data)
                    running_mean = running_mean.view_as(normalized.running_mean)
                    running_var = running_var.view_as(normalized.running_var)
                    # Copy the loaded weights to model
                    normalized.bias.data.copy_(biases)
                    normalized.weight.data.copy_(weight)
                    normalized.running_mean.copy_(running_mean)
                    normalized.running_var.copy_(running_var)
                else:
                    # Get the number of biases
                    num_bias = convolution.bias.numel()
                    # Load the weights
                    biases = torch.from_numpy(weights[start: start + num_bias])
                    start += num_bias
                    # Reshape the loaded weights according to the dimension of the model weight
                    biases = biases.view_as(convolution.bias.data)
                    # Copy the loaded weights to model
                    convolution.bias.data.copy_(biases)
                # Load the weights for convolutional layer
                num_weight = convolution.weight.numel()
                weight = torch.from_numpy(weights[start: start + num_weight])
                start += num_weight
                # Cast the loaded weights into the dimension of the model weight
                weight = weight.view_as(convolution.weight.data)
                # Copy the loaded weights to model
                convolution.weight.data.copy_(weight)

    def save_weight(self, weight_file: str):
        def to_cpu(_x: tensor):
            if _x.is_cuda:
                return torch.FloatTensor(_x.size()).copy_(_x)
            else:
                return _x

        file = open(weight_file, "wb")
        # Attach the header at the top of the file
        self.header[3] = self.seen
        self.header.numpy().tofile(file)
        # Save the weight to file directly
        blocks = self.blocks[1:]
        for i, block in enumerate(blocks):
            type = block['type']
            if type == "convolutional":
                try:
                    batch_norm = int(block['batch_normalize'])
                except:
                    batch_norm = 0
                convolution = self.modules[i][0]
                if batch_norm > 0:
                    normalized = self.modules[i][1]
                    to_cpu(normalized.bias.data).numpy().tofile(file)
                    to_cpu(normalized.weight.data).numpy().tofile(file)
                    to_cpu(normalized.running_mean).numpy().tofile(file)
                    to_cpu(normalized.running_var).numpy().tofile(file)
                else:
                    to_cpu(convolution.bias.data).numpy().tofile(file)
                to_cpu(convolution.weight.data).numpy().tofile(file)
