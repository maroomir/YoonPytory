import pickle

def parse_cifar10_trainer(strRootDir: str,
                          strMode: str = "alexnet"  # alexnet, resnet, unet, vgg
                          ):
    with open(strRootDir)