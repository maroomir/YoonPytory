import sys
import librosa
import matplotlib.pyplot
import numpy
import scipy
import soundfile
from scipy.fftpack import fft
from sklearn.preprocessing import minmax_scale


class YoonSpeech:
    __signal: list
    samplingRate: int
    fftCount: int = 512
    melOrder: int = 24
    mfccOrder: int = 13
    windowLength: float = 0.02
    shiftLength: float = 0.005

    def __str__(self):
        return "SIGNAL LENGTH : {0}, SAMPLING RATE : {1}".format(len(self.__signal), self.samplingRate)

    def __init__(self,
                 strFileName: str = None,
                 pSignal: list = None,
                 nSamplingRate: int = 48000,
                 nFFTCount: int = 512,
                 nMelOrder: int = 24,
                 nMFCCOrder: int = 13,
                 dWindowLength: float = 0.02,
                 dShiftLength: float = 0.005
                 ):
        self.samplingRate = nSamplingRate
        self.fftCount = nFFTCount
        self.melOrder = nMelOrder
        self.mfccOrder = nMFCCOrder
        self.windowLength = dWindowLength
        self.shiftLength = dShiftLength
        if strFileName is not None:
            self.load_sound_file(strFileName)
        elif pSignal is not None:
            self.__signal = pSignal
        else:
            self.__signal = None

    def __copy__(self):
        return YoonSpeech(pSignal=self.__signal, nSamplingRate=self.samplingRate)

    def load_sound_file(self, strFileName: str):
        self.__signal, self.samplingRate = librosa.load(strFileName, self.samplingRate)

    def save_sound_file(self, strFileName: str):
        soundfile.write(strFileName, self.__signal, self.samplingRate)

    def resampling(self, nTargetRate: int):
        pListResampling = librosa.resample(self.__signal, self.samplingRate, nTargetRate)
        return YoonSpeech(pSignal=pListResampling, nSamplingRate=nTargetRate)

    def crop(self, dStartTime: float, dEndTime: float):
        iStart, iEnd = int(dStartTime * self.samplingRate), int(dEndTime * self.samplingRate)
        return YoonSpeech(pSignal=self.__signal[iStart, iEnd], nSamplingRate=self.samplingRate)

    def scaling(self, dMin=-0.99999, dMax=0.99999):
        pSignal = minmax_scale(self.__signal, feature_range=(dMin, dMax))
        return YoonSpeech(pSignal=pSignal, nSamplingRate=self.samplingRate)

    def show_time_signal(self):
        # Init graph
        pFigure = matplotlib.pyplot.figure(figsize=(14, 8))
        pGraph = pFigure.add_subplot(211)
        pGraph.set_title('Raw Speech Signal')
        pGraph.set_xlabel('Time (sec)')
        pGraph.set_ylabel('Amplitude')
        pGraph.grid(True)
        # Set graph per time sample
        nCountTime = len(self.__signal)
        listUnitX = numpy.linspace(0, nCountTime / self.samplingRate, nCountTime)
        pGraph.set_xlim(listUnitX.min(), listUnitX.max())
        pGraph.set_ylim(self.__signal.min() * 1.4, self.__signal.max() * 1.4)
        pGraph.plot(listUnitX, self.__signal)
        # Show graph
        matplotlib.pyplot.show()

    @staticmethod
    def show_mfcc(pArrayData: numpy.ndarray):
        # Init Graph
        matplotlib.pyplot.title('MFCC')
        matplotlib.pyplot.ylabel('The number of Coefficients')
        matplotlib.pyplot.xlabel('The number of frames')
        # Set graph per Frequency
        matplotlib.pyplot.imshow(pArrayData[:, 1:].transpose(), cmap='jet', origin='lower', aspect='auto')
        matplotlib.pyplot.colorbar()
        # Show graph
        matplotlib.pyplot.show()

    @staticmethod
    def show_mel_spectrum(pArrayData: numpy.ndarray):
        # Init Graph
        matplotlib.pyplot.title('Power Mel Filterbanks Spectogram')
        matplotlib.pyplot.ylabel('The number of Mel Filterbanks')
        matplotlib.pyplot.xlabel('The number of frames')
        # Set graph per Frequency
        matplotlib.pyplot.imshow(pArrayData.transpose(), cmap='jet', origin='lower', aspect='auto')
        matplotlib.pyplot.colorbar()
        # Show graph
        matplotlib.pyplot.show()

    # Compute Fast Fourier Transformation
    def __fft(self):
        pArrayWindow = self.__signal * numpy.hanning(len(self.__signal))
        nSizeFFT = pow(2, int(numpy.log2(len(pArrayWindow))) + 1)  # Pow of 2
        pArrayFrequency = fft(pArrayWindow, nSizeFFT)
        return pArrayFrequency

    # Compute Short Time Fourier Transformation
    def __stft(self, nFFTCount: int, dWindowLength: float, dShiftLength: float):
        nCountWindowSample = int(dWindowLength * self.samplingRate)
        nCountStepSample = int(dShiftLength * self.samplingRate)
        nCountFrames = int(numpy.floor((len(self.__signal) - nCountWindowSample) / float(nCountStepSample)) + 1)
        pArrayWindow = numpy.hanning(nCountWindowSample)
        pListSourceSection = []
        for iStep in range(0, nCountFrames):
            pListSourceSection.append(
                pArrayWindow * self.__signal[
                               iStep * nCountStepSample: iStep * nCountStepSample + nCountWindowSample])
        pFrame = numpy.stack(pListSourceSection)
        return numpy.fft.rfft(pFrame, n=nFFTCount, axis=1)

    # Combine features depending on the context window setup
    @staticmethod
    def __context_window(pFeature: numpy.ndarray, nSizeContext: int):
        if nSizeContext < 2:
            Exception("Context window length is too short")
        nLeft = int(nSizeContext / 2)
        nRight = nSizeContext - nLeft
        pListResult = []
        for i in range(nLeft, len(pFeature) - nRight):
            pArray = numpy.concatenate((pFeature[i - nLeft: i + nRight]), axis=-1)
            pListResult.append(pArray)
        return numpy.vstack(pListResult)

    # Compute magnitude and Log-magnitude spectrum
    def get_log_magnitude(self,
                          strFFTType: str = 'fft'):
        if strFFTType == 'fft':
            pArrayFreq = self.__fft()
            nSizeHalf = int(len(pArrayFreq) / 2)  # Use only Half
            pArrayMagnitude = abs(pArrayFreq[0:nSizeHalf])
            return 20 * numpy.log10(pArrayMagnitude)
        elif strFFTType == 'stft':
            pArrayFreq = self.__stft(self.fftCount, self.windowLength, self.shiftLength)
            pArrayMagnitude = abs(pArrayFreq)
            return 20 * numpy.log10(pArrayMagnitude + 1.0e-10)
        else:
            print('Wrong Fourier transform type : {}'.format(strFFTType))
            raise StopIteration

    def get_feature(self, strFeatureType: str):
        if strFeatureType == "mel":
            return self.scaling(-0.9999, 0.9999).get_log_mel_spectrum()
        elif strFeatureType == "mfcc":
            return self.scaling(-0.9999, 0.9999).get_mfcc()
        elif strFeatureType == "deltas":
            return self.get_mfcc_deltas(bContext=True, nLengthContext=10)
        else:
            Exception("Feature type is not correct")

    def get_log_mel_spectrum(self):
        pArrayFreqSignal = self.__stft(self.fftCount, self.windowLength, self.shiftLength)
        pArrayMagnitude = abs(pArrayFreqSignal) ** 2
        pArrayMelFilter = librosa.filters.mel(self.samplingRate, self.fftCount, self.melOrder)
        pArrayMelSpectrogram = numpy.matmul(pArrayMagnitude, pArrayMelFilter.transpose())  # Multiply Matrix
        return 10 * numpy.log10(pArrayMelSpectrogram)

    def get_log_mel_deltas(self,
                           bContext: bool = False,
                           nLengthContext: int = 10):
        # Perform a short-time Fourier Transform
        nShiftCount = int(self.shiftLength * self.samplingRate)
        nWindowCount = int(self.windowLength * self.samplingRate)
        pArrayFreqSignal = librosa.core.stft(self.__signal, n_fft=self.fftCount, hop_length=nShiftCount,
                                             win_length=nWindowCount)
        pArrayFeature = abs(pArrayFreqSignal).transpose()
        # Estimate either log mep-spectrum
        pArrayMelFilter = librosa.filters.mel(self.samplingRate, n_fft=self.fftCount, n_mels=self.melOrder)
        pPowerFeature = pArrayFeature ** 2
        pArrayFeature = numpy.matmul(pPowerFeature, pArrayMelFilter.transpose())
        pArrayFeature = 10 * numpy.log10(pArrayFeature + numpy.array(sys.float_info.epsilon))  # feature + Epsilon
        pArrayDelta1 = librosa.feature.delta(pArrayFeature)
        pArrayDelta2 = librosa.feature.delta(pArrayFeature, order=2)
        pArrayResult = numpy.concatenate(pArrayFeature, pArrayDelta1, pArrayDelta2)
        if bContext:
            pArrayResult = self.__context_window(pArrayResult, nLengthContext)
        return pArrayResult

    def get_mfcc(self):
        pArrayLogMelSpectrogram = self.get_log_mel_spectrum()
        pArrayMFCC = scipy.fftpack.dct(pArrayLogMelSpectrogram, axis=-1,
                                       norm='ortho')  # Discreate cosine transformation
        return pArrayMFCC[:, :self.melOrder]

    def get_mfcc_deltas(self,
                        bContext: bool = False,
                        nLengthContext: int = 10):
        # Perform a short-time Fourier Transform
        nShiftCount = int(self.shiftLength * self.samplingRate)
        nWindowCount = int(self.windowLength * self.samplingRate)
        pArrayFreqSignal = librosa.core.stft(self.__signal, n_fft=self.fftCount, hop_length=nShiftCount,
                                             win_length=nWindowCount)
        pArrayFeature = abs(pArrayFreqSignal).transpose()
        # Estimate either log mep-spectrum
        pArrayMelFilter = librosa.filters.mel(self.samplingRate, n_fft=self.fftCount, n_mels=self.melOrder)
        pPowerFeature = pArrayFeature ** 2
        pArrayFeature = numpy.matmul(pPowerFeature, pArrayMelFilter.transpose())
        pArrayFeature = 10 * numpy.log10(pArrayFeature + numpy.array(sys.float_info.epsilon))  # feature + Epsilon
        pArrayFeature = scipy.fftpack.dct(pArrayFeature, axis=-1, norm='ortho')
        pArrayDelta1 = librosa.feature.delta(pArrayFeature)
        pArrayDelta2 = librosa.feature.delta(pArrayFeature)
        pArrayResult = numpy.concatenate((pArrayFeature, pArrayDelta1, pArrayDelta2), axis=-1)
        if bContext:
            pArrayResult = self.__context_window(pArrayResult, nLengthContext)
        return pArrayResult
