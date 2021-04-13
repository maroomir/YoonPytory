import numpy
import soundfile
import librosa
import pickle
from scipy.fftpack import fft


class YoonSpeech:
    timeSignals: list = None
    logMelSpectrogram: list = None
    MFCCs: list = None
    samplingRate: int = 0

    def __str__(self):
        return ""

    def load_wave_file(self, strFileName: str):
        self.timeSignals, self.samplingRate = soundfile.read(strFileName)
        print("Load Wave file to Playtime {0:0.2f} seconds, Sampling rate {1} Hz".format(
            len(self.timeSignals) / self.samplingRate, self.samplingRate))

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

    def __fft(self):
        pListSignalWindowed = self.timeSignals * numpy.hanning(len(self.timeSignals))
        nSizeFFT = pow(2, int(numpy.log2(len(pListSignalWindowed))) + 1)  # Pow of 2
        pListFreqSignal = fft(pListSignalWindowed, nSizeFFT)
        return pListFreqSignal

    # Compute Short Time Fourier Transformation
    def __stft(self, nFFTCount: int, dWindowLength: float, dShiftLength: float):
        nCountWindowSample = int(dWindowLength * self.samplingRate)
        nCountStepSample = int(dShiftLength * self.samplingRate)
        nCountFrames = int(numpy.floor((len(self.timeSignals) - nCountWindowSample) / float(nCountStepSample)) + 1)
        pListWindow = numpy.hanning(len(nCountWindowSample))
        pListSourceSection = []
        for iStep in range(0, nCountFrames):
            pListSourceSection.append(
                pListWindow * self.timeSignals[iStep * nCountStepSample: iStep * nCountStepSample + nCountWindowSample])
        pFrame = numpy.stack(pListSourceSection)
        return numpy.fft.rfft(pFrame, n=nFFTCount, axis=1)