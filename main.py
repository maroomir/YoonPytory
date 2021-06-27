import yoonpytory
import yoonimage
import yoonspeech
import yoonspeech.recognition
import yoonimage.classification


def process_test_dir():
    # Get Direction list
    list_dir = yoonpytory.dir2D.get_clock_directions()
    print("Number of clock direction : {0}".format(len(list_dir)))
    print("Directions : {0}".format(list_dir))
    # Go direction per "Ordering"
    direction = list_dir[0]
    vector = yoonpytory.vector2D.direction_vector(direction)
    for i in range(10):
        print("DIR {0} = ".format(i) + direction.__str__())
        print("VECTOR {0} = ".format(i) + vector.__str__())
        direction += "order"
        vector += "order"
    # Add direction directly
    print("LAST DIR = " + direction.__str__())
    direction += yoonpytory.dir2D.RIGHT
    print("ADD DIR = " + direction.__str__())


def process_test_vector():
    vector1 = yoonpytory.vector2D(10, 15)
    vector2 = yoonpytory.vector2D(13, 19)
    print("VEC 1 = {0}, VEC 2 = {1}, DISTANCE = {2}".format(vector1.to_tuple(), vector2.to_tuple(),
                                                            vector1.distance(vector2)))
    scale_vector = vector1.scale(2, 2)
    move_vector = yoonpytory.vector2D(10, 10, nStepX=5, nStepY=5)
    print("Scale VEC 1 = " + scale_vector.__str__())
    # Move vector to direction
    for i in range(10):
        print("MOVE {0} = ".format(i) + move_vector.__str__())
        move_vector += yoonpytory.dir2D.TOP_RIGHT


def process_test_rect():
    vector = yoonpytory.vector2D(10, 10, nStepX=5, nStepY=5)
    list_vector = [vector]
    for i_dir in yoonpytory.dir2D.get_square_directions():
        print("INSERT VECTOR = " + vector.__str__())
        vector += i_dir
        print("DIR TO = " + i_dir.__str__())
        list_vector.append(vector)
    rect1 = yoonpytory.rect2D(list=list_vector)
    rect2 = yoonpytory.rect2D(x=0, y=0, width=15, height=15)
    print("RECT FROM TUPLES = " + rect1.__str__())
    print("OBJECT RECT = " + rect2.__str__())
    print("SUM = " + (rect1 + rect2).__str__())


def process_test_line():
    # Move vector to direction
    move_vector = yoonpytory.vector2D(5, 5, nStepX=5, nStepY=5)
    list_vector = []
    for i in range(10):
        print("MOVE {0} = ".format(i) + move_vector.__str__())
        list_vector.append(move_vector.__copy__())
        move_vector += yoonpytory.dir2D.TOP_RIGHT
    line1 = yoonpytory.line2D(list_vector[0], list_vector[1], list_vector[2], list_vector[3], list_vector[4])
    line2 = yoonpytory.line2D(list=list_vector)
    other_vector = yoonpytory.vector2D(1, -1)
    print(line1.__str__())
    print(line2.__str__())
    print("DIST = {}".format(line1.distance(other_vector)))
    print((line1 + line2).__str__())


def process_test_yolo():
    # Run network
    net_param = yoonimage.YoloNet()
    net_param.load_modern_net(strWeightFile="./data/yolo/yolov3.weights", strConfigFile="./data/yolo/yolov3.cfg",
                              strNamesFile="./data/yolo/coco.names")
    image = yoonimage.image(strFileName="./data/yolo/input1.bmp")
    obj_list = yoonimage.detection(image, net_param, pSize=yoonimage.YOLO_SIZE_NORMAL,
                                   dScale=yoonimage.YOLO_SCALE_ONE_ZERO_PER_8BIT)
    result = yoonimage.remove_noise(obj_list)
    yoonimage.draw_detection_result(result, image, net_param)


def process_single_layer_perception():
    net = yoonpytory.neuron()
    net.load_source(strFileName='./data/slp/twoGaussians.npz')
    net.load_weight(strFileName='./data/slp/weight.npz')
    net.train(nCountEpoch=2000, bInitWeight=True, bRunTest=False)
    net.process()
    net.save_weight(strFileName='./data/slp/weight.npz')


def process_multi_layer_perception():
    net = yoonpytory.network()
    net.load_source(strFileName='./data/mlp/spirals.npz')
    net.load_weight(strFileName='./data/mlp/weights.npz')
    # net.train(nCountEpoch=1000, nSizeLayer=100, nOrder=10, bInitWeight=False, bRunTest=False)
    net.process()
    net.show_plot()
    net.save_weight(strFileName='./data/mlp/weights.npz')


def process_speech():
    speech = yoonspeech.speech(dWindowLength=0.004, dShiftLength=0.001)
    speech.load_sound_file(strFileName='./data/speech/2021451143.wav')
    speech.show_time_signal()
    mel_frames = speech.get_log_mel_spectrum()
    speech.show_mel_spectrum(mel_frames)
    mfcc_frames = speech.get_mfcc()
    speech.show_mfcc(mfcc_frames)
    print(speech.__str__())


def process_speaker_recognition():
    sampling_rate = 16000
    window_length = 0.025
    shift_length = 0.01
    # Train
    pTrainData, pTestData = yoonspeech.parse_librispeech_trainer(
        strRootDir='./data/speech/LibriSpeech/dev-clean',
        nSamplingRate=sampling_rate, dRatioTrain=0.8,
        dWindowLength=window_length, dShiftLength=shift_length,
        strFeatureType="mfcc")
    yoonspeech.speakerRecognition.gmm.train(pTrainData, pTestData, strModelPath='./data/speech/GMM.mdl')
    # Speaker recognition with gmm
    speech = yoonspeech.speech(nSamplingRate=sampling_rate, dWindowLength=window_length, dShiftLength=shift_length)
    speech.load_sound_file(strFileName='./data/speech/2021451143.wav')
    yoonspeech.speakerRecognition.gmm.recognition(speech, strModelPath='./data/speech/GMM.mdl',
                                                  strFeatureType="mfcc")
    speech_female = yoonspeech.speech(nSamplingRate=sampling_rate, dWindowLength=window_length,
                                      dShiftLength=shift_length)
    speech_female.load_sound_file(strFileName='./data/speech/yeseul.wav')
    yoonspeech.speakerRecognition.gmm.recognition(speech_female, strModelPath='./data/speech/GMM.mdl',
                                                  strFeatureType="mfcc")


def process_speaker_recognition_with_torch():
    sampling_rate = 16000
    window_length = 0.025
    shift_length = 0.01
    epoch = 100
    # Train
    pTrainData, pTestData = yoonspeech.parse_librispeech_trainer(
        strRootDir='./data/speech/LibriSpeech/dev-clean',
        nSamplingRate=sampling_rate, dRatioTrain=0.8,
        dWindowLength=window_length, dShiftLength=shift_length,
        strFeatureType="deltas")
    yoonspeech.speakerRecognition.dvector.train(epoch, pTrainData=pTrainData, pValidationData=pTestData,
                                                strModelPath='./data/speech/model_opt.pth',
                                                bInitEpoch=False)
    yoonspeech.speakerRecognition.dvector.test(pTestData, './data/speech/model_opt.pth')
    # Speaker recognition with d-vector
    class_count = pTestData.class_count
    speech = yoonspeech.speech(nSamplingRate=sampling_rate, dWindowLength=window_length, dShiftLength=shift_length)
    speech.load_sound_file(strFileName='./data/speech/2021451143.wav')
    label = yoonspeech.speakerRecognition.dvector.recognition(pSpeech=speech, nCountClass=class_count,
                                                              strModelPath='./data/speech/model_opt.pth')
    print("The speaker is estimated : " + pTestData.names[label])


def process_speech_recognition_with_ctc():
    sampling_rate = 16000
    window_length = 0.025
    shift_length = 0.01
    epoch = 100
    # Train
    train_data, test_data = yoonspeech.parse_librispeech_trainer(
        strRootDir='./data/speech/LibriSpeech/dev-clean',
        nSamplingRate=sampling_rate, dRatioTrain=0.8,
        dWindowLength=window_length, dShiftLength=shift_length,
        strFeatureType="deltas", nContextSize=1)  # Do not use the context for precious learning
    yoonspeech.recognition.ctc.train(epoch, pTrainData=train_data, pValidationData=test_data,
                                     strModelPath='./data/speech/ctc_opt.pth')


def process_speech_recognition_with_las():
    sampling_rate = 16000
    window_length = 0.025
    shift_length = 0.01
    epoch = 100
    # Train
    train_data, eval_data = yoonspeech.parse_librispeech_trainer(
        strRootDir='./data/speech/LibriSpeech/dev-clean',
        nSamplingRate=sampling_rate, dRatioTrain=0.8,
        dWindowLength=window_length, dShiftLength=shift_length,
        strFeatureType="deltas", nContextSize=1)  # Do not use the context for precious learning
    yoonspeech.recognition.las.train(epoch, pTrainData=train_data, pValidationData=eval_data,
                                     strModelPath='./data/speech/las_opt.pth')


def process_image_segmentation(mode="resnet"):
    class_count, train_data, eval_data = yoonimage.parse_cifar10_trainer(
        strRootDir='./data/image/cifar-10', dRatioTrain=0.8, strMode=mode)
    epoch = 1000
    train_data.draw_dataset(5, 5, "name")
    eval_data.draw_dataset(5, 5, "name")
    if mode == "alexnet":
        yoonimage.classification.alexnet.train(epoch, pTrainData=train_data, pEvalData=eval_data,
                                               nCountClass=class_count, strModelPath='./data/image/alex_opt.pth')
    elif mode == "vgg":
        yoonimage.classification.vgg.train(epoch, pTrainData=train_data, pEvalData=eval_data,
                                           nCountClass=class_count, strModelPath='./data/image/vgg_opt.pth')
    elif mode == "resnet":
        yoonimage.classification.resnet.train(epoch, pTrainData=train_data, pEvalData=eval_data,
                                              nCountClass=class_count, strModelPath='./data/image/res_opt.pth')


if __name__ == '__main__':
    # process_test_dir()
    # process_test_vector()
    # process_test_rect()
    # process_test_line()
    # process_test_yolo()
    # process_single_layer_perception()
    # process_multi_layer_perception()
    # process_speech()
    # process_speaker_recognition()
    # process_speaker_recognition_with_torch()
    # process_speech_recognition_with_ctc()
    # process_speech_recognition_with_las()
    process_image_segmentation()
