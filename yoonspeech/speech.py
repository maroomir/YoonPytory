import numpy
import matplotlib.pyplot
import soundfile
import librosa
import pickle
import scipy
from scipy.fftpack import fft


class YoonSpeech:
    timeSignals: list
    logMelSpectrogram: numpy.ndarray
    MFCCs: numpy.ndarray
    samplingRate: int

    def __str__(self):
        return "LOG MEL : {0}, MFCC : {1}".format(self.logMelSpectrogram.shape(), self.MFCCs.shape())

    def __init__(self,
                 strWavFileName: str = None,
                 pListTimeSignal: list = None,
                 nSamplingRate: int = 48000):
        if strWavFileName is not None:
            self.load_wave_file(strWavFileName)
        elif pListTimeSignal is not None:
            self.timeSignals = pListTimeSignal
            self.samplingRate = nSamplingRate
            self.logMelSpectrogram = self._log_mel_spectrogram()
            self.MFCCs = self._mel_frequency_cepstrogram_coefficient()
        else:
            self.timeSignals = None
            self.samplingRate = 0
            self.logMelSpectrogram = None
            self.MFCCs = None

    def __copy__(self):
        return YoonSpeech(pListTimeSignal=self.timeSignals, nSamplingRate=self.samplingRate)

    def load_wave_file(self, strFileName: str):
        self.timeSignals, self.samplingRate = librosa.load(strFileName)
        print("Load Wave file to Playtime {0:0.2f} seconds, Sampling rate {1} Hz".format(
            len(self.timeSignals) / self.samplingRate, self.samplingRate))
        self.logMelSpectrogram = self._log_mel_spectrogram()
        self.MFCCs = self._mel_frequency_cepstrogram_coefficient()

    def save_wave_file(self, strFileName: str):
        soundfile.write(strFileName, self.timeSignals, self.samplingRate)

    def load_log_mel_spectrogram(self, strFileName: str):
        with open(strFileName, "rb") as pFile:
            self.logMelSpectrogram = pickle.load(pFile)
            print("Load Log-Mel Spectrogram : {} components".format(numpy.shape(self.logMelSpectrogram)))

    def save_log_mel_spectrogram(self, strFileName: str):
        with open(strFileName, "wb") as pFile:
            pickle.dump(self.logMelSpectrogram, pFile)

    def load_mfcc(self, strFileName: str):
        with open(strFileName, "rb") as pFile:
            self.MFCCs = pickle.load(pFile)
            print("Load MFCC : {} components".format(numpy.shape(self.MFCCs)))

    def save_mfcc(self, strFileName: str):
        with open(strFileName, "wb") as pFile:
            pickle.dump(self.MFCCs, pFile)

    def resampling(self, nTargetRate: int):
        pListResampling = librosa.resample(self.timeSignals, self.samplingRate, nTargetRate)
        return YoonSpeech(pListTimeSignal=pListResampling, nSamplingRate=nTargetRate)

    def crop(self, dStartTime: float, dEndTime: float):
        iStart, iEnd = int(dStartTime * self.samplingRate), int(dEndTime * self.samplingRate)
        return YoonSpeech(pListTimeSignal=self.timeSignals[iStart, iEnd], nSamplingRate=self.samplingRate)

    def show_time_signal(self):
        # Init graph
        pFigure = matplotlib.pyplot.figure(figsize=(14, 8))
        pGraph = pFigure.add_subplot(211)
        pGraph.set_title('Raw Speech Signal')
        pGraph.set_xlabel('Time (sec)')
        pGraph.set_ylabel('Amplitude')
        pGraph.grid(True)
        # Set graph per time sample
        nCountTime = len(self.timeSignals)
        listUnitX = numpy.linspace(0, nCountTime / self.samplingRate, nCountTime)
        pGraph.set_xlim(listUnitX.min(), listUnitX.max())
        pGraph.set_ylim(self.timeSignals.min() * 1.4, self.timeSignals.max() * 1.4)
        pGraph.plot(listUnitX, self.timeSignals)
        # Show graph
        matplotlib.pyplot.show()

    def show_mfcc(self):
        # Init Graph
        matplotlib.pyplot.title('MFCC')
        matplotlib.pyplot.ylabel('The number of Coefficients')
        matplotlib.pyplot.xlabel('The number of frames')
        # Set graph per Frequency
        matplotlib.pyplot.imshow(self.MFCCs.transpose(), cmap='jet', origin='lower', aspect='auto')
        matplotlib.pyplot.colorbar()
        # Show graph
        matplotlib.pyplot.show()

    def show_log_mel_spectrogram(self):
        # Init Graph
        matplotlib.pyplot.title('Power Mel Filterbanks Spectogram')
        matplotlib.pyplot.ylabel('The number of Mel Filterbanks')
        matplotlib.pyplot.xlabel('The number of frames')
        # Set graph per Frequency
        matplotlib.pyplot.imshow(self.logMelSpectrogram.transpose(), cmap='jet', origin='lower', aspect='auto')
        matplotlib.pyplot.colorbar()
        # Show graph
        matplotlib.pyplot.show()

    # Compute magnitude and Log-magnitude spectrum
    def _log_magnitude(self,
                       strFFTType: str = 'fft',
                       nFFTCount: int = 512,
                       dWindowLength: float = 0.02,
                       dShiftLength: float = 0.005):
        if strFFTType == 'fft':
            pArrayFreq = self.__fft()
            nSizeHalf = int(len(pArrayFreq) / 2)  # Use only Half
            pArrayMagnitude = abs(pArrayFreq[0:nSizeHalf])
            return 20 * numpy.log10(pArrayMagnitude)
        elif strFFTType == 'stft':
            pArrayFreq = self.__stft(nFFTCount, dWindowLength, dShiftLength)
            pArrayMagnitude = abs(pArrayFreq)
            return 20 * numpy.log10(pArrayMagnitude + 1.0e-10)
        else:
            print('Wrong Fourier transform type : {}'.format(strFFTType))
            raise StopIteration

    def _log_mel_spectrogram(self,
                             nFFTCount: int = 512,
                             nMelOrder: int = 24,
                             dWindowLength: float = 0.02,
                             dShiftLength: float = 0.005):
        pArrayFreqSignal = self.__stft(nFFTCount, dWindowLength, dShiftLength)
        pArrayMagnitude = abs(pArrayFreqSignal) ** 2
        pArrayMelFilter = librosa.filters.mel(self.samplingRate, nFFTCount, nMelOrder)
        pArrayMelSpectrogram = numpy.matmul(pArrayMagnitude, pArrayMelFilter.transpose())  # Multiply Matrix
        return 10 * numpy.log10(pArrayMelSpectrogram)

    def _mel_frequency_cepstrogram_coefficient(self,
                                               nFFTCount: int = 512,
                                               nMelOrder: int = 24,
                                               nMFCCOrder: int = 13,
                                               dWindowLength: float = 0.02,
                                               dShiftLength: float = 0.005):
        pArrayLogMelSpectrogram = self._log_mel_spectrogram(nFFTCount, nMelOrder, dWindowLength, dShiftLength)
        pArrayMFCC = scipy.fftpack.dct(pArrayLogMelSpectrogram, axis=-1, norm='ortho')  # Discreate cosine transformation
        return pArrayMFCC[:, :nMFCCOrder]

    def __fft(self):
        pArrayWindow = self.timeSignals * numpy.hanning(len(self.timeSignals))
        nSizeFFT = pow(2, int(numpy.log2(len(pArrayWindow))) + 1)  # Pow of 2
        pArrayFrequency = fft(pArrayWindow, nSizeFFT)
        return pArrayFrequency

    # Compute Short Time Fourier Transformation
    def __stft(self, nFFTCount: int, dWindowLength: float, dShiftLength: float):
        nCountWindowSample = int(dWindowLength * self.samplingRate)
        nCountStepSample = int(dShiftLength * self.samplingRate)
        nCountFrames = int(numpy.floor((len(self.timeSignals) - nCountWindowSample) / float(nCountStepSample)) + 1)
        pArrayWindow = numpy.hanning(nCountWindowSample)
        pListSourceSection = []
        for iStep in range(0, nCountFrames):
            pListSourceSection.append(
                pArrayWindow * self.timeSignals[iStep * nCountStepSample: iStep * nCountStepSample + nCountWindowSample])
        pFrame = numpy.stack(pListSourceSection)
        return numpy.fft.rfft(pFrame, n=nFFTCount, axis=1)
