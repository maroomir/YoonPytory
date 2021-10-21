import collections
import os
import yoonspeech
from os.path import splitext, basename

from tqdm import tqdm

from yoonspeech.data import YoonDataset
from yoonspeech.data import YoonObject
from yoonspeech.speech import YoonSpeech


def get_phoneme_list(file_path: str):
    with open(file_path, 'r') as file:
        paths = file.read().split('\n')[:-1]
    paths = [tag.split(' ')[-1] for tag in paths]
    paths = list(set(paths))
    return paths


def get_phoneme_dict(file_path: str):
    with open(file_path, 'r') as pFile:
        paths = pFile.read().split('\n')[:-1]
    dic = {}
    for tag in paths:
        if tag.split(' ')[0] == 'q':
            pass
        else:
            dic[tag.split(' ')[0]] = tag.split(' ')[-1]
    return dic


def parse_librispeech_trainer(root_dir: str,
                              file_type: str = '.flac',
                              sample_len: int = 1000,
                              sample_rate: int = 16000,
                              fft_count: int = 512,
                              mel_order: int = 24,
                              mfcc_order: int = 13,
                              context_size: int = 10,
                              win_len: float = 0.025,
                              shift_len: float = 0.01,
                              feature_type: str = "mfcc",
                              train_rate: float = 0.8,
                              mode: str = "dvector"  # dvector, gmm, ctc, las
                              ):

    def get_line_in_trans(file_path, id):
        with open(file_path) as pFile:
            lines = pFile.read().lower().split('\n')[:-1]
        for line in lines:
            if id in line:
                line = line.replace(id + ' ', "")
                return line

    def make_speech_buffer(file_path):
        speech = YoonSpeech(sample_rate=sample_rate, context_size=context_size,
                            fft_count=fft_count, mel_order=mel_order, mfcc_order=mfcc_order,
                            win_len=win_len, shift_len=shift_len)
        speech.load_sound_file(file_path)
        return speech

    feature_file_dic = collections.defaultdict(list)
    trans_file_dic = collections.defaultdict(dict)
    trans_files = []
    test_files = []
    # Extract file names
    for root, dir_, file_paths in tqdm(os.walk(root_dir)):
        i = 0
        for path in file_paths:
            if splitext(path)[1] == file_type:
                id_ = splitext(path)[0].split('-')[0]
                feature_file_dic[id_].append(os.path.join(root, path))
                i += 1
                if i > sample_len:
                    break
            elif splitext(path)[1] == ".txt":  # Recognition the words
                id_, part = splitext(path)[0].split('-')
                part = part.replace(".trans", "")
                trans_file_dic[id_][part] = os.path.join(root, path)
    # Listing test and train dataset
    for i, file_paths in feature_file_dic.items():
        trans_files.extend(file_paths[:int(len(file_paths) * train_rate)])
        test_files.extend(file_paths[int(len(file_paths) * train_rate):])
    # Labeling speakers for Speaker recognition
    speaker_dic = {}
    speakers = list(feature_file_dic.keys())
    num_speakers = len(speakers)
    for i in range(num_speakers):
        speaker_dic[speakers[i]] = i
    # Transform data dictionary
    train_dataset = YoonDataset()
    eval_dataset = YoonDataset()
    for path in trans_files:
        basename_ = splitext(basename(path))[0]
        id_, part = basename_.split('-')[0], basename_.split('-')[1]
        word = get_line_in_trans(trans_file_dic[id_][part], basename_)
        speech = make_speech_buffer(path)
        obj = YoonObject(id=int(speaker_dic[id_]), name=id_, word=word, type=feature_type, speech=speech)
        train_dataset.append(obj)
    for path in test_files:
        basename_ = splitext(basename(path))[0]
        id_, part = basename_.split('-')[0], basename_.split('-')[1]
        word = get_line_in_trans(trans_file_dic[id_][part], basename_)
        speech = make_speech_buffer(path)
        obj = YoonObject(id=int(speaker_dic[id_]), name=id_, word=word, type=feature_type, speech=speech)
        eval_dataset.append(obj)
    print("Length of Train = {}".format(train_dataset.__len__()))
    print("Length of Test = {}".format(eval_dataset.__len__()))
    if mode == "dvector" or mode == "gmm":
        output_dim = num_speakers
    elif mode == "ctc" or mode == "las":
        output_dim = yoonspeech.DEFAULT_PHONEME_COUNT
    else:
        raise ValueError("Unsupported parsing mode")
    return output_dim, train_dataset, eval_dataset


def parse_librispeech_tester(strRootDir: str,
                             strFileType: str = '.flac',
                             nCountSample: int = 1000,
                             nSamplingRate: int = 16000,
                             nFFTCount: int = 512,
                             nMelOrder: int = 24,
                             nMFCCOrder: int = 13,
                             nContextSize: int = 10,
                             dWindowLength: float = 0.025,
                             dShiftLength: float = 0.01,
                             strFeatureType: str = "mfcc",
                             strMode: str = "dvector"  # dvector, gmm, ctc, las
                             ):
    pDicFile = collections.defaultdict(list)
    pListTestFile = []
    # Extract file names
    for strRoot, strDir, pListFileName in tqdm(os.walk(strRootDir)):
        iCount = 0
        for strFileName in pListFileName:
            if splitext(strFileName)[1] == strFileType:
                strID = splitext(strFileName)[0].split('-')[0]
                pDicFile[strID].append(os.path.join(strRoot, strFileName))
                iCount += 1
                if iCount > nCountSample:
                    break
    # Listing test and train dataset
    for i, pListFileName in pDicFile.items():
        pListTestFile.extend(pListFileName[:int(len(pListFileName))])
    # Labeling speakers for PyTorch Training
    pDicLabel = {}
    pListSpeakers = list(pDicFile.keys())
    nSpeakersCount = len(pListSpeakers)
    for i in range(nSpeakersCount):
        pDicLabel[pListSpeakers[i]] = i
    # Transform data dictionary
    pDataTest = YoonDataset()
    for strFileName in pListTestFile:
        strID = splitext(basename(strFileName))[0].split('-')[0]
        pSpeech = YoonSpeech(strFileName=strFileName, nSamplingRate=nSamplingRate, strFeatureType=strFeatureType,
                             nContextSize=nContextSize,
                             nFFTCount=nFFTCount, nMelOrder=nMelOrder, nMFCCOrder=nMFCCOrder,
                             dWindowLength=dWindowLength, dShiftLength=dShiftLength)
        pObject = YoonObject(nID=int(pDicLabel[strID]), strName=strID, strType=strFeatureType, pSpeech=pSpeech)
        pDataTest.append(pObject)
    if strMode == "dvector" or strMode == "gmm":
        nDimOutput = nSpeakersCount
    elif strMode == "ctc" or strMode == "las":
        nDimOutput = yoonspeech.DEFAULT_PHONEME_COUNT
    else:
        raise ValueError("Unsupported parsing mode")
    return nDimOutput, pDataTest
