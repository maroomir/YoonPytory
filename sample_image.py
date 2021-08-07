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


def process_drop():
    count, dataset = yoonimage.parse_root(strRootDir='./data/image/Drops')
    for data_object in dataset:
        results = yoonimage.find_blobs(pSourceImage=data_object.image,
                                       nThreshold=100, nMaxCount=20)
        for result_object in results:
            data_object.image.draw_rectangle(result_object.region, yoonimage.COLOR_YELLOW)
            data_object.image.show_image()
        print("count = " + str(len(results)))


def process_glass():
    count, dataset = yoonimage.parse_root(strRootDir='./data/image/Glass')
    for data_object in dataset:
        image = data_object.image
        results = yoonimage.find_lines(pSourceImage=image,
                                       nThresh1=50, nThresh2=200)
        for result_object in results:
            image.draw_line(result_object.region, yoonimage.COLOR_YELLOW)
        print("count = " + str(len(results)))
        image.show_image()


if __name__ == '__main__':
    print("Select the sample process")
    print("1. alexnet")
    print("2. vgg")
    print("3. resnet")
    print("4. drop")
    print("5. Glass")
    process = input(">>")
    process = process.lower()
    if process == "1" or process == "alexnet":
        process_segmentation("alexnet")
    elif process == "2" or process == "vgg":
        process_segmentation("vgg")
    elif process == "3" or process == "resnet":
        process_segmentation("resnet")
    elif process == "4" or process == "drop":
        process_drop()
    elif process == "5" or process == "glass":
        process_glass()
