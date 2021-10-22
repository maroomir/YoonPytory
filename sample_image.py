import yoonimage
import yoonimage.classification


def process_segmentation(mode="resnet"):
    class_count, transform, train_data, eval_data = yoonimage.parse_cifar10_trainer(
        root='./data/image/cifar-10', train_ratio=0.8)
    epoch = 1000
    train_data.draw_dataset(5, 5, "name")
    eval_data.draw_dataset(5, 5, "name")
    if mode == "alexnet":
        yoonimage.classification.alexnet.train(epoch, train_data=train_data, eval_data=eval_data, transform=transform,
                                               num_class=class_count, model_path='./data/image/alex_opt.pth')
    elif mode == "vgg":
        yoonimage.classification.vgg.train(epoch, train_data=train_data, eval_data=eval_data, transform=transform,
                                           num_class=class_count, model_path='./data/image/vgg_opt.pth')
    elif mode == "resnet":
        yoonimage.classification.resnet.train(epoch, train_data=train_data, eval_data=eval_data, transform=transform,
                                              num_class=class_count, model_path='./data/image/res_opt.pth')


def process_drop():
    count, dataset = yoonimage.parse_root(root='./data/image/Drops')
    for i in range(0, count):
        image = dataset[i].image
        results = yoonimage.find_blobs(source=image,
                                       thresh=100, max_count=20)
        for result_obj in results:
            image.draw_rectangle(result_obj.region, yoonimage.COLOR_YELLOW)
            image.show_image()
        print("count = " + str(len(results)))


def process_glass():
    count, dataset = yoonimage.parse_root(root='./data/image/Glass')
    for i in range(0, count):
        image = dataset[i].image
        results = yoonimage.find_lines(source=image,
                                       thresh1=50, thresh2=150,
                                       max_count=30)
        for result_obj in results:
            image.draw_line(result_obj.region, yoonimage.COLOR_YELLOW)
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
