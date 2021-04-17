import numpy
import matplotlib.pyplot
import soundfile
import librosa
import pickle
import scipy
from scipy.fftpack import fft


class YoonSpeech:
    __timeSignal: list
    __frameData: numpy.ndarray
    dataType: str
    samplingRate: int
    ffTCount: int
    melOrder: int
    mfccOrder: int
    windowLength: float
    shiftLength: float

    def __str__(self):
        return "SIGNAL LENGTH : {0}, FRAME SHAPE : {1}".format(len(self.__timeSignal), self.__frameData.shape)

    def __init__(self,
                 strFileName: str = None,
                 pListTimeSignal: list = None,
                 nSamplingRate: int = 48000,
                 strType: str = "mfcc",
                 nFFTCount: int = 512,
                 nMelOrder: int = 24,
                 nMFCCOrder: int = 13,
                 dWindowLength: float = 0.02,
                 dShiftLength: float = 0.005):
        self.samplingRate = nSamplingRate
        self.dataType = strType
        self.ffTCount = nFFTCount
        self.melOrder = nMelOrder
        self.mfccOrder = nMFCCOrder
        self.windowLength = dWindowLength
        self.shiftLength = dShiftLength
        if strFileName is not None:
            self.load_sound_file(strFileName)
        elif pListTimeSignal is not None:
            self.__timeSignal = pListTimeSignal
            if self.dataType == "mel":
                self.__frameData = self.__log_mel_spectrogram()
            elif self.dataType == "mfcc":
                self.__frameData = self.__mel_frequency_cepstrogram_coefficient()
            else:
                Exception("Transform type is abnormal : ", self.dataType)
        else:
            self.__timeSignal = None
            self.__frameData = None

    def __copy__(self):
        return YoonSpeech(pListTimeSignal=self.__timeSignal, nSamplingRate=self.samplingRate)

    def load_sound_file(self, strFileName: str):
        self.__timeSignal, self.samplingRate = librosa.load(strFileName, self.samplingRate)
        if self.dataType == "mel":
            self.__frameData = self.__log_mel_spectrogram()
        elif self.dataType == "mfcc":
            self.__frameData = self.__mel_frequency_cepstrogram_coefficient()
        else:
            Exception("Transform type is abnormal : ", self.dataType)
        print("Load sound file to Playtime {0:0.2f} seconds, Sampling rate {1} Hz"
              .format(len(self.__timeSignal) / self.samplingRate, self.samplingRate))

    def save_sound_file(self, strFileName: str):
        soundfile.write(strFileName, self.__timeSignal, self.samplingRate)

    def load_frame_data(self, strFileName: str):
        with open(strFileName, "rb") as pFile:
            self.__frameData = pickle.load(pFile)
            print("Load {} Data : {} components".format(self.dataType, numpy.shape(self.__frameData)))

    def save_frame_data(self, strFileName: str):
        with open(strFileName, "wb") as pFile:
            pickle.dump(self.__frameData, pFile)

    def resampling(self, nTargetRate: int):
        pListResampling = librosa.resample(self.__timeSignal, self.samplingRate, nTargetRate)
        return YoonSpeech(pListTimeSignal=pListResampling, nSamplingRate=nTargetRate)

    def crop(self, dStartTime: float, dEndTime: float):
        iStart, iEnd = int(dStartTime * self.samplingRate), int(dEndTime * self.samplingRate)
        return YoonSpeech(pListTimeSignal=self.__timeSignal[iStart, iEnd], nSamplingRate=self.samplingRate)

    def show_time_signal(self):
        # Init graph
        pFigure = matplotlib.pyplot.figure(figsize=(14, 8))
        pGraph = pFigure.add_subplot(211)
        pGraph.set_title('Raw Speech Signal')
        pGraph.set_xlabel('Time (sec)')
        pGraph.set_ylabel('Amplitude')
        pGraph.grid(True)
        # Set graph per time sample
        nCountTime = len(self.__timeSignal)
        listUnitX = numpy.linspace(0, nCountTime / self.samplingRate, nCountTime)
        pGraph.set_xlim(listUnitX.min(), listUnitX.max())
        pGraph.set_ylim(self.__timeSignal.min() * 1.4, self.__timeSignal.max() * 1.4)
        pGraph.plot(listUnitX, self.__timeSignal)
        # Show graph
        matplotlib.pyplot.show()

    def show_frame_data(self):
        if self.dataType == "mfcc":
            # Init Graph
            matplotlib.pyplot.title('MFCC')
            matplotlib.pyplot.ylabel('The number of Coefficients')
            matplotlib.pyplot.xlabel('The number of frames')
            # Set graph per Frequency
            matplotlib.pyplot.imshow(self.__frameData[:, 1:].transpose(), cmap='jet', origin='lower', aspect='auto')
            matplotlib.pyplot.colorbar()
            # Show graph
            matplotlib.pyplot.show()
        elif self.dataType == "mel":
            # Init Graph
            matplotlib.pyplot.title('Power Mel Filterbanks Spectogram')
            matplotlib.pyplot.ylabel('The number of Mel Filterbanks')
            matplotlib.pyplot.xlabel('The number of frames')
            # Set graph per Frequency
            matplotlib.pyplot.imshow(self.__frameData.transpose(), cmap='jet', origin='lower', aspect='auto')
            matplotlib.pyplot.colorbar()
            # Show graph
            matplotlib.pyplot.show()

    # Compute magnitude and Log-magnitude spectrum
    def __log_magnitude(self, strFFTType: str = 'fft'):
        if strFFTType == 'fft':
            pArrayFreq = self.__fft()
            nSizeHalf = int(len(pArrayFreq) / 2)  # Use only Half
            pArrayMagnitude = abs(pArrayFreq[0:nSizeHalf])
            return 20 * numpy.log10(pArrayMagnitude)
        elif strFFTType == 'stft':
            pArrayFreq = self.__stft(self.ffTCount, self.windowLength, self.shiftLength)
            pArrayMagnitude = abs(pArrayFreq)
            return 20 * numpy.log10(pArrayMagnitude + 1.0e-10)
        else:
            print('Wrong Fourier transform type : {}'.format(strFFTType))
            raise StopIteration

    def __log_mel_spectrogram(self):
        pArrayFreqSignal = self.__stft(self.ffTCount, self.windowLength, self.shiftLength)
        pArrayMagnitude = abs(pArrayFreqSignal) ** 2
        pArrayMelFilter = librosa.filters.mel(self.samplingRate, self.ffTCount, self.melOrder)
        pArrayMelSpectrogram = numpy.matmul(pArrayMagnitude, pArrayMelFilter.transpose())  # Multiply Matrix
        return 10 * numpy.log10(pArrayMelSpectrogram)

    def __mel_frequency_cepstrogram_coefficient(self):
        pArrayLogMelSpectrogram = self.__log_mel_spectrogram()
        pArrayMFCC = scipy.fftpack.dct(pArrayLogMelSpectrogram, axis=-1,
                                       norm='ortho')  # Discreate cosine transformation
        return pArrayMFCC[:, :self.mfccOrder]

    def __fft(self):
        pArrayWindow = self.__timeSignal * numpy.hanning(len(self.__timeSignal))
        nSizeFFT = pow(2, int(numpy.log2(len(pArrayWindow))) + 1)  # Pow of 2
        pArrayFrequency = fft(pArrayWindow, nSizeFFT)
        return pArrayFrequency

    # Compute Short Time Fourier Transformation
    def __stft(self, nFFTCount: int, dWindowLength: float, dShiftLength: float):
        nCountWindowSample = int(dWindowLength * self.samplingRate)
        nCountStepSample = int(dShiftLength * self.samplingRate)
        nCountFrames = int(numpy.floor((len(self.__timeSignal) - nCountWindowSample) / float(nCountStepSample)) + 1)
        pArrayWindow = numpy.hanning(nCountWindowSample)
        pListSourceSection = []
        for iStep in range(0, nCountFrames):
            pListSourceSection.append(
                pArrayWindow * self.__timeSignal[
                               iStep * nCountStepSample: iStep * nCountStepSample + nCountWindowSample])
        pFrame = numpy.stack(pListSourceSection)
        return numpy.fft.rfft(pFrame, n=nFFTCount, axis=1)

    @staticmethod
    def listing(pList: list, strType: str):
        pListResult = []
        for pSpeech in pList:
            pDic = {"time", pSpeech.__timeSignal,
                    "frame", pSpeech.__frameData}
            pListResult.append(pDic[strType])
        return pListResult
