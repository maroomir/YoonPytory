import yoonspeech
import yoonspeech.recognition


def process_transfer():
    speech = yoonspeech.speech(dWindowLength=0.004, dShiftLength=0.001)
    speech.load_sound_file(file_name='./data/speech/2021451143.wav')
    speech.show_time_signal()
    mel_frames = speech.get_log_mel_spectrum()
    speech.show_mel_spectrum(mel_frames)
    mfcc_frames = speech.get_mfcc()
    speech.show_mfcc(mfcc_frames)
    print(speech.__str__())


def process_gmm():  # Speaker Recognition
    sampling_rate = 16000
    window_length = 0.025
    shift_length = 0.01
    # Train
    pTrainData, pTestData = yoonspeech.parse_librispeech_trainer(
        root_dir='./data/speech/LibriSpeech/dev-clean',
        sample_rate=sampling_rate, train_rate=0.8,
        win_len=window_length, shift_len=shift_length,
        feature_type="mfcc")
    yoonspeech.speakerRecognition.gmm.train(pTrainData, pTestData, strModelPath='./data/speech/GMM.mdl')
    # Speaker recognition with gmm
    speech = yoonspeech.speech(nSamplingRate=sampling_rate, dWindowLength=window_length, dShiftLength=shift_length)
    speech.load_sound_file(file_name='./data/speech/2021451143.wav')
    yoonspeech.speakerRecognition.gmm.recognition(speech, strModelPath='./data/speech/GMM.mdl',
                                                  strFeatureType="mfcc")
    speech_female = yoonspeech.speech(nSamplingRate=sampling_rate, dWindowLength=window_length,
                                      dShiftLength=shift_length)
    speech_female.load_sound_file(file_name='./data/speech/yeseul.wav')
    yoonspeech.speakerRecognition.gmm.recognition(speech_female, strModelPath='./data/speech/GMM.mdl',
                                                  strFeatureType="mfcc")


def process_dvector():  # Speaker Recognition
    sampling_rate = 16000
    window_length = 0.025
    shift_length = 0.01
    epoch = 100
    # Train
    pTrainData, pTestData = yoonspeech.parse_librispeech_trainer(
        root_dir='./data/speech/LibriSpeech/dev-clean',
        sample_rate=sampling_rate, train_rate=0.8,
        win_len=window_length, shift_len=shift_length,
        feature_type="deltas")
    yoonspeech.speakerRecognition.dvector.train(epoch, pTrainData=pTrainData, pValidationData=pTestData,
                                                strModelPath='./data/speech/model_opt.pth',
                                                bInitEpoch=False)
    yoonspeech.speakerRecognition.dvector.test(pTestData, './data/speech/model_opt.pth')
    # Speaker recognition with d-vector
    class_count = pTestData.class_count
    speech = yoonspeech.speech(nSamplingRate=sampling_rate, dWindowLength=window_length, dShiftLength=shift_length)
    speech.load_sound_file(file_name='./data/speech/2021451143.wav')
    label = yoonspeech.speakerRecognition.dvector.recognition(pSpeech=speech, nCountClass=class_count,
                                                              strModelPath='./data/speech/model_opt.pth')
    print("The speaker is estimated : " + pTestData.names[label])


def process_ctc():  # Speech Recognition
    sampling_rate = 16000
    window_length = 0.025
    shift_length = 0.01
    epoch = 100
    # Train
    train_data, test_data = yoonspeech.parse_librispeech_trainer(
        root_dir='./data/speech/LibriSpeech/dev-clean',
        sample_rate=sampling_rate, train_rate=0.8,
        win_len=window_length, shift_len=shift_length,
        feature_type="deltas", context_size=1)  # Do not use the context for precious learning
    yoonspeech.recognition.ctc.train(epoch, pTrainData=train_data, pValidationData=test_data,
                                     strModelPath='./data/speech/ctc_opt.pth')


def process_las():  # Speech Recognition
    sampling_rate = 16000
    window_length = 0.025
    shift_length = 0.01
    epoch = 100
    # Train
    train_data, eval_data = yoonspeech.parse_librispeech_trainer(
        root_dir='./data/speech/LibriSpeech/dev-clean',
        sample_rate=sampling_rate, train_rate=0.8,
        win_len=window_length, shift_len=shift_length,
        feature_type="deltas", context_size=1)  # Do not use the context for precious learning
    yoonspeech.recognition.las.train(epoch, pTrainData=train_data, pValidationData=eval_data,
                                     strModelPath='./data/speech/las_opt.pth')


if __name__ == '__main__':
    print("Select the sample process")
    print("1. Transfer")
    print("2. GMM")  # Speaker Recognition
    print("3. DVector")  # Speaker Recognition
    print("4. CTC")  # Speech Recognition
    print("5. LAS")  # Speech Recognition
    process = input(">>")
    process = process.lower()
    if process == "1" or "transfer":
        process_transfer("alexnet")
    elif process == "2" or "gmm":
        process_gmm()
    elif process == "3" or "dvector":
        process_dvector()
    elif process == "4" or "ctc":
        process_ctc()
    elif process == "5" or "las":
        process_las()