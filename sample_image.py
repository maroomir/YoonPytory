import yoonimage
import yoonimage.classification

def process_segmentation(mode="resnet"):
    class_count, transform, train_data, eval_data = yoonimage.parse_cifar10_trainer(
        strRootDir='./data/image/cifar-10', dRatioTrain=0.8, strMode=mode)
    epoch = 1000
    train_data.draw_dataset(5, 5, "name")
    eval_data.draw_dataset(5, 5, "name")
    if mode == "alexnet":
        yoonimage.classification.alexnet.train(epoch, pTrainData=train_data, pEvalData=eval_data, pTransform=transform,
                                               nCountClass=class_count, strModelPath='./data/image/alex_opt.pth')
    elif mode == "vgg":
        yoonimage.classification.vgg.train(epoch, pTrainData=train_data, pEvalData=eval_data, pTransform=transform,
                                           nCountClass=class_count, strModelPath='./data/image/vgg_opt.pth')
    elif mode == "resnet":
        yoonimage.classification.resnet.train(epoch, pTrainData=train_data, pEvalData=eval_data, pTransform=transform,
                                              nCountClass=class_count, strModelPath='./data/image/res_opt.pth')


if __name__ == '__main__':
    print("Select the sample process")
    print("1. alexnet")
    print("2. vgg")
    print("3. resnet")
    process = input(">>")
    process = process.lower()
    if process == "1" or "alexnet":
        process_segmentation("alexnet")
    elif process == "2" or "vgg":
        process_segmentation("vgg")
    elif process == "3" or "resnet":
        process_segmentation("resnet")