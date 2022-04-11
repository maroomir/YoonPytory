import math
import os.path
from datetime import datetime

import h5py
import numpy
import scipy
import scipy.io
import skimage.metrics
import torch
import torch.nn
import torch.nn.functional
from numpy import ndarray
from torch import tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomTransform(object):
    def __init__(self,
                 mean=0.5,
                 std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self,
                 image: ndarray,
                 target: ndarray = None):
        def to_tensor(x: ndarray):  # Height X Width X Channel
            if numpy.iscomplexobj(x):
                x = numpy.stack((x.real, x.imag), axis=-1)
            x = x.transpose((2, 0, 1)).astype(numpy.float32)  # Channel X Height X Width
            return torch.from_numpy(x)

        def z_normalize(x: tensor):
            mask_func = x > 0
            mean = x[mask_func].mean()
            std = x[mask_func].std()
            return torch.mul((x - mean) / std, mask_func.float())

        def minmax_normalize(x: tensor):
            max = x.max()
            min = x.min()
            return (x - min) / (max - min)

        def normalize(x: tensor):
            return (x - self.mean) / self.std

        def pixel_compress(x: tensor):
            max_func = x > 255
            min_func = x < 0
            x[max_func] = 255
            x[min_func] = 0
            return x / 255.0

        input_ = minmax_normalize(to_tensor(image))
        if target is not None:
            target_ = minmax_normalize(to_tensor(target))
            return input_, target_
        else:
            return input_


class ConvDataset(Dataset):
    def __init__(self,
                 file_path,
                 transform=None,
                 mode_="train",  # train, eval, test
                 train_ratio=0.8):
        self.transform = transform
        self.mode = mode_
        # Initial the H5PY Inputs
        self.len = 0
        self.height = 0
        self.width = 0
        self.channel = 0
        self.input_data = None
        self.label_data = None
        self.load_dataset(file_path, ratio=train_ratio)

    def load_dataset(self, file_path: str, ratio: float):
        file = h5py.File(file_path)
        input_ = numpy.array(file['input'], dtype=numpy.float32)  # 216 X 384 X 384 X 3
        try:
            label = numpy.array(file['label'], dtype=numpy.float32)  # 216 X 384 X 384 X 1
        except:
            label = None
            print("Label data is not contained")
        self.len, self.height, self.width, self.channel = input_.shape
        num_train = int(self.len * ratio)
        if self.mode == "train":
            self.input_data = input_[:num_train, :, :, :]
            self.label_data = label[:num_train, :, :, :]
        elif self.mode == "eval":
            self.input_data = input_[num_train:, :, :, :]
            self.label_data = label[num_train:, :, :, :]
        elif self.mode == "test":
            self.input_data = input_
        else:
            raise Exception("Data mode is not compatible")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, item):
        if self.mode == "test":
            return self.transform(self.input_data[item])
        else:
            return self.transform(self.input_data[item], self.label_data[item])


class CustomLoss(Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.prediction = None
        self.target = None
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, predict: tensor, target: tensor):
        mse_loss = self.mse_loss(predict, target)
        l1_loss = self.l1_loss(predict, target)
        self.prediction = predict.detach().cpu()
        self.target = target.detach().cpu()
        ssim = 1 - self.ssim_score()
        return mse_loss + l1_loss + ssim

    def psnr_score(self):
        psnr = 0.0
        for batch in range(self.prediction.shape[0]):
            predict = self.prediction[batch].numpy()
            target = self.target[batch].numpy()
            psnr += skimage.metrics.peak_signal_noise_ratio(target, predict)
        return psnr / self.prediction.shape[0]

    def ssim_score(self):
        ssim = 0.0
        for batch in range(self.prediction.shape[0]):
            predict = self.prediction[batch].numpy()
            target = self.target[batch].numpy()
            ssim += skimage.metrics.structural_similarity(target, predict)
        return ssim / self.prediction.shape[0]

    def dice_coefficient(self, smooth=1e-4):
        predict = self.prediction.contiguous().view(-1)
        target = self.target.contiguous().view(-1)
        intersection = (predict * target).sum()
        coefficient = (2.0 * intersection + smooth) / (predict.sum() + target.sum() + smooth)
        return coefficient


def save_labels(output: ndarray, file_path):
    scipy.io.savemat(file_path, mdict={'y_pred': output})
    print("Save output files completed!")


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


class DiscriminateBlock(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        super(DiscriminateBlock, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(output_dim),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: tensor):
        return self.network(x)


class Discriminator(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel: int,
                 num_depth: int):
        super(Discriminator, self).__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(input_dim, channel, kernel_size=4, stride=2, padding=1),
                                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))])
        for i in range(num_depth - 1):
            self.blocks += [DiscriminateBlock(channel, channel * 2)]
            channel *= 2
        self.blocks += [torch.nn.Sequential(
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(channel, output_dim, kernel_size=4, stride=2, padding=1, bias=False))]

    def forward(self, x: tensor, target: tensor):
        result = torch.cat((x, target), dim=1)
        for _, block in enumerate(self.blocks):
            result = block(result)
        return result


class GeneratorUNet(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel: int,
                 num_depth: int,
                 dropout: float = 0.3):
        super(GeneratorUNet, self).__init__()
        # Init Encoders and Decoders
        self.encoders = torch.nn.ModuleList([ConvolutionBlock(input_dim, channel, dropout)])
        for i in range(num_depth - 1):
            self.encoders += [ConvolutionBlock(channel, channel * 2, dropout)]
            channel *= 2
        self.down_sampler = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.worker = ConvolutionBlock(channel, channel * 2, dropout)
        self.decoders = torch.nn.ModuleList()
        self.up_samplers = torch.nn.ModuleList()
        for i in range(num_depth - 1):
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
        width_pad = floor_ceil((width_margin - width) / 2)
        height_pad = floor_ceil((height_margin - height) / 2)
        x = torch.nn.functional.pad(x, width_pad + height_pad)
        return x, (height_pad, width_pad, height_margin, width_margin)

    def __unpadding(self, x, height_pad, width_pad, height_margin, width_margin):
        return x[...,
               height_pad[0]:height_margin - height_pad[1],
               width_pad[0]:width_margin - width_pad[1]]

    def forward(self, x: tensor):
        x, pad_option = self.__padding(x)
        stacks = []
        result = x
        # Apply down sampling layers
        for i, encoder in enumerate(self.encoders):
            result = encoder(result)
            stacks.append(result)
            result = self.down_sampler(result)
        result = self.worker(result)
        # Apply up sampling layers
        for sampler, decoder in zip(self.up_samplers, self.decoders):
            attached = stacks.pop()
            result = sampler(result)
            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]  # left, right, top, bottom
            if result.shape[-1] != attached.shape[-1]:
                padding[1] = 1  # Padding right
            if result.shape[-2] != attached.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                result = torch.nn.functional.pad(result, padding, "reflect")
            result = torch.cat([result, attached], dim=1)
            result = decoder(result)
        stacks.clear()  # To Memory Optimizing
        result = self.__unpadding(result, *pad_option)
        return result


def __trace_in__(message: str):
    now = datetime.now()
    file_path = "LOG_PIX2PIX_" + now.strftime("%Y-%m-%d") + '.txt'
    with open(file_path, mode='a') as pFile:
        pFile.write("[" + now.strftime("%H:%M:%S") + "] " + message + "\n")


def __process_train(epoch: int, data_loader: DataLoader,
                    generator: GeneratorUNet, discriminator: Discriminator,
                    generator_criterion: CustomLoss, discriminator_criterion: Module,
                    generator_optimizer: Adam, discriminator_optimizer: Adam):
    def __train_generator(_input: tensor,
                          _target: tensor,
                          outside=0.01):
        generator_optimizer.zero_grad()
        fake = generator(_input).to(device)
        fake_predict = discriminator(_input, fake).to(device)
        # Reshape the tensor data of use
        fake = fake.squeeze(axis=1)  # reshape : (batch, 384, 384)
        fake_predict = fake_predict.squeeze(axis=1)
        _target = _target.squeeze(axis=1)
        # Set-up the ground truths (Shape like the prediction size)
        pass_ = torch.ones(fake_predict.size(), requires_grad=True).type(torch.FloatTensor).to(device)
        # Compute the network accuracy
        generator_loss = generator_criterion(fake, _target)  # Loss the generator
        discriminator_loss = discriminator_criterion(fake_predict, pass_)  # Loss the pass rate of the fake image
        loss = generator_loss + (discriminator_loss * outside)
        psnr = generator_criterion.psnr_score()
        ssim = generator_criterion.ssim_score()
        # Perform backpropagation to update GENERATOR parameters
        loss.backward()
        generator_optimizer.step()
        # Fix the CUDA Out of Memory problem
        del fake
        del fake_predict
        torch.cuda.empty_cache()
        return loss.item(), psnr, ssim

    def __train_discriminator(_input: tensor,
                              _target: tensor):
        discriminator_optimizer.zero_grad()
        target_fake = generator(_input).to(device).to(device)
        predict_real = discriminator(_input, _target).to(device)
        predict_fake = discriminator(_input, target_fake).to(device)
        # Reshape the tensor data of use
        predict_real = predict_real.squeeze(axis=1)
        predict_fake = predict_fake.squeeze(axis=1)
        # Set-up the ground truths (Shape like the prediction size)
        pass_ = torch.ones(predict_real.size(), requires_grad=True).type(torch.FloatTensor).to(device)
        ng = torch.zeros(predict_real.size(), requires_grad=True).type(torch.FloatTensor).to(device)
        # Compute the Real Pass or NG rate
        real_ng_loss = discriminator_criterion(predict_fake, ng)
        real_ok_loss = discriminator_criterion(predict_real, pass_)
        loss = real_ng_loss + real_ok_loss
        # Perform backpropagation to update GENERATOR parameters
        loss.backward()
        discriminator_optimizer.step()
        # Fix the CUDA Out of Memory problem
        del target_fake
        del predict_real
        del predict_fake
        torch.cuda.empty_cache()
        return loss.item(), real_ok_loss.item(), real_ng_loss.item()

    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Perform a training using the defined network
    generator.train()
    discriminator.train()
    # Warp the iterable Data Loader with TQDM
    bar = tqdm(enumerate(data_loader))
    message = ""
    for i, (input_, target) in bar:
        # Move data and label to device
        input_ = input_.type(torch.FloatTensor).to(device)  # Shape : (batch, 3, 384, 384)
        target = target.type(torch.FloatTensor).to(device)  # Shape : (batch, 1, 384, 384)
        # Pass the input data through the GENERATOR architecture
        generator_loss, psnr, ssim = __train_generator(input_, target, outside=0.01)
        # Pass the input data through the GENERATOR architecture
        discriminator_loss, real_ok, real_ng = __train_discriminator(input_, target)
        # Perform backpropagation to update DISCRIMINATOR parameters
        message = "Train Epoch:{:3d} [{}/{} {:.2f}%], " \
                  "Generator={:.4f}, PSNR={:.4f}, SSIM={:.4f}, " \
                  "RealOK={:.4f}, RealNG={:.4f}".format(epoch, i + 1, len(data_loader),
                                                        100.0 * ((i + 1) / len(data_loader)),
                                                        generator_loss, psnr, ssim, real_ok, real_ng)
        bar.set_description(message)
    # trace the last message
    __trace_in__(message)


def __process_evaluate(model: GeneratorUNet, data_loader: DataLoader, criterion: CustomLoss):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Perform an evaluation using the defined network
    model.eval()
    # Warp the iterable Data Loader with TQDM
    bar = tqdm(enumerate(data_loader))
    sample_len = 0
    total_loss = 0
    total_psnr = 0.0
    total_ssim = 0.0
    message = ""
    with torch.no_grad():
        for i, (input_, target) in bar:
            # Move data and label to device
            input_ = input_.type(torch.FloatTensor).to(device)
            target = target.type(torch.FloatTensor).to(device)
            # Pass the input data through the defined network architecture
            output = model(input_).squeeze(axis=1)  # reshape : (batch, 384, 384)
            target = target.squeeze(axis=1)  # reshape : (batch, 384, 384)
            # Compute a loss function
            loss = criterion(output, target).to(device)
            total_loss += loss.item() * len(target)
            # Compute network accuracy
            total_psnr += criterion.psnr_score() * len(target)
            total_ssim += criterion.ssim_score() * len(target)
            sample_len += len(target)
            message = "Eval {}/{} {:.2f}%, Loss={:.4f}, PSNR={:.4f}, SSIM={:.4f}". \
                format(i + 1, len(data_loader), 100.0 * ((i + 1) / len(data_loader)),
                       total_loss / sample_len, total_psnr / sample_len, total_ssim / sample_len)
            bar.set_description(message)
    # trace the last message
    __trace_in__(message)
    # Fix the CUDA Out of Memory problem
    del output
    del loss
    torch.cuda.empty_cache()
    return total_loss / sample_len


def train(epoch: int,
          file_path: str,
          generator_path: str = None,
          discriminator_path: str = None,
          channel=8,
          num_depth=4,
          batch_size=1,
          num_worker=0,  # 0: CPU / 4 : GPU
          dropout=0.3,
          decay=0.5,
          init_epoch=False,
          ):
    def learning_func(step):
        return 1.0 - max(0, step - epoch * (1 - decay)) / (decay * epoch + 1)

    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define the training and testing data-set
    train_set = ConvDataset(file_path=file_path, transform=CustomTransform(), mode_="train", train_ratio=0.8)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_worker, pin_memory=True)
    valid_set = ConvDataset(file_path=file_path, transform=CustomTransform(), mode_="eval", train_ratio=0.8)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False,
                              num_workers=num_worker, pin_memory=True)
    # Define a network model
    generator = GeneratorUNet(input_dim=3, output_dim=1, channel=channel, num_depth=num_depth,
                              dropout=dropout).to(device)  # T1, T2, GRE(3) => STIR(1)
    discriminator = Discriminator(input_dim=4, output_dim=1, channel=64,
                                  num_depth=num_depth).to(device)  # Input(3), Target(1) => BOOL(1)
    # Set the optimizer with adam
    generator_optimizer = torch.optim.Adam(generator.parameters())
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    # Set the scheduler to control the learning rate
    generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, lr_lambda=learning_func)
    discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(discriminator_optimizer, lr_lambda=learning_func)
    # Set the Loss Function
    generator_criterion = CustomLoss()
    discriminator_criterion = torch.nn.MSELoss()
    # Load pre-trained model
    start = 0
    print("Directory of the generator model: {}".format(generator_path))
    if generator_path is not None and os.path.exists(generator_path) and init_epoch is False:
        model_data = torch.load(generator_path, map_location=device)
        start = model_data['epoch']
        generator.load_state_dict(model_data['model'])
        generator_optimizer.load_state_dict(model_data['optimizer'])
        print("## Successfully load the Generator!")
    print("Directory of the discriminator model: {}".format(discriminator_path))
    if discriminator_path is not None and os.path.exists(discriminator_path) and init_epoch is False:
        model_data = torch.load(discriminator_path, map_location=device)
        start = model_data['epoch'] if model_data['epoch'] < start else start
        discriminator.load_state_dict(model_data['model'])
        discriminator_optimizer.load_state_dict(model_data['optimizer'])
        print("## Successfully load the Discriminator!")
    # Train and Test Repeat
    min_loss = 10000.0
    for epoch in range(start, epoch + 1):
        # Train the network
        __process_train(epoch, data_loader=train_loader, generator=generator, discriminator=discriminator,
                        generator_criterion=generator_criterion, discriminator_criterion=discriminator_criterion,
                        generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
        # Test the network
        loss = __process_evaluate(model=generator, data_loader=valid_loader, criterion=generator_criterion)
        # Change the learning rate
        generator_scheduler.step()
        discriminator_scheduler.step()
        # Rollback the model when loss is NaN
        if math.isnan(loss):
            if generator_path is not None and os.path.exists(generator_path):
                # Reload the best model and decrease the learning rate
                model_data = torch.load(generator_path, map_location=device)
                generator.load_state_dict(model_data['model'])
                optimizer_data = model_data['optimizer']
                optimizer_data['param_groups'][0]['lr'] /= 2  # Decrease the learning rate by 2
                generator_optimizer.load_state_dict(optimizer_data)
                print("## Rollback the Generator with half learning rate!")
            if discriminator_path is not None and os.path.exists(discriminator_path):
                # Reload the best model and decrease the learning rate
                model_data = torch.load(discriminator_path, map_location=device)
                discriminator.load_state_dict(model_data['model'])
                optimizer_data = model_data['optimizer']
                optimizer_data['param_groups'][0]['lr'] /= 2  # Decrease the learning rate by 2
                discriminator_optimizer.load_state_dict(optimizer_data)
                print("## Rollback the Discriminator with half learning rate!")
        # Save the optimal model
        elif loss < min_loss:
            min_loss = loss
            torch.save({'epoch': epoch, 'model': generator.state_dict(),
                        'optimizer': generator_optimizer.state_dict()}, generator_path)
            torch.save({'epoch': epoch, 'model': discriminator.state_dict(),
                        'optimizer': discriminator_optimizer.state_dict()}, discriminator_path)
        elif epoch % 100 == 0:
            torch.save({'epoch': epoch, 'model': generator.state_dict(),
                        'optimizer': generator_optimizer.state_dict()}, 'gen_{}epoch.pth'.format(epoch))
            torch.save({'epoch': epoch, 'model': discriminator.state_dict(),
                        'optimizer': discriminator_optimizer.state_dict()}, 'disc_{}epoch.pth'.format(epoch))


def test(file_path: str,
         model_path: str,
         channel=8,
         num_depth=4,
         num_worker=0,  # 0: CPU / 4 : GPU
         dropout=0.3):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("{} device activation".format(device.__str__()))
    # Define a network model
    model = GeneratorUNet(input_dim=3, output_dim=1, channel=channel, num_depth=num_depth,
                          dropout=dropout).to(device)
    model_data = torch.load(model_path, map_location=device)
    model.load_state_dict(model_data['model'])
    model.eval()
    print("Successfully load the Model in path")
    # Define the validation data-set
    test_set = ConvDataset(file_path=file_path, transform=CustomTransform(), mode_="test")
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False,
                             num_workers=num_worker, pin_memory=True)
    bar = tqdm(test_loader)
    results = []
    with torch.no_grad():
        for input_ in bar:
            input_ = input_.to(device)
            result = model(input_)
            result = torch.argmax(result, dim=1)
            for i in range(result.shape[0]):  # Attach per batch
                results.append(result[i].detach().cpu().numpy())
    # Save the result to mat
    save_labels(numpy.array(results), 'ContrastConversion.mat')


if __name__ == '__main__':
    mode = 'train'
    if mode == 'all':
        train(epoch=1000,
              file_path='contrast_conversion_train_dataset.mat',
              generator_path='model_gen.pth',
              discriminator_path='model_disc.pth',
              channel=64,  # 8 >= VRAM 9GB / 4 >= VRAM 6.5GB
              num_depth=4,
              batch_size=2,
              num_worker=0,  # 0: CPU / 4 : GPU
              dropout=0.3,
              decay=0.5,
              init_epoch=False)
        test(file_path='contrast_conversion_train_dataset.mat',
             model_path='model_gen.pth',
             channel=64,  # 8 : colab / 4 : RTX2070
             num_depth=4,
             num_worker=0,  # 0: CPU / 4 : GPU
             dropout=0)
    elif mode == 'train':
        train(epoch=3000,
              file_path='contrast_conversion_train_dataset.mat',
              generator_path='model_gen.pth',
              discriminator_path='model_disc.pth',
              channel=64,  # 8 >= VRAM 9GB / 4 >= VRAM 6.5GB
              num_depth=4,
              batch_size=2,
              num_worker=0,  # 0: CPU / 4 : GPU
              dropout=0.3,
              decay=0.5,
              init_epoch=False)
    elif mode == 'test':
        test(file_path='contrast_conversion_train_dataset.mat',
             model_path='model_gen.pth',
             channel=64,  # 8 : colab / 4 : RTX2070
             num_depth=4,
             num_worker=0,  # 0: CPU / 4 : GPU
             dropout=0)
    else:
        pass
