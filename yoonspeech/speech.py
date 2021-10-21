import sys
import librosa
import matplotlib.pyplot
import numpy
import scipy
import soundfile
from numpy import ndarray
from scipy.fftpack import fft
from sklearn.preprocessing import minmax_scale


class YoonSpeech:
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    __signal: list
    sampling_rate: int
    fft_count: int = 512
    mel_order: int = 24
    mfcc_order: int = 13
    window_length: float = 0.02
    shift_length: float = 0.005
    context_size: int = 10
    """

    def __str__(self):
        return "SIGNAL LENGTH : {0}, SAMPLING RATE : {1}".format(len(self.__signal), self.sample_rate)

    def __init__(self,
                 signal: ndarray = None,
                 sample_rate: int = 48000,
                 fft_count: int = 512,
                 mel_order: int = 24,
                 mfcc_order: int = 13,
                 context_size: int = 10,
                 win_len: float = 0.02,
                 shift_len: float = 0.005
                 ):
        self.sample_rate = sample_rate
        self.fft_count = fft_count
        self.mel_order = mel_order
        self.mfcc_order = mfcc_order
        self.window_length = win_len
        self.shift_length = shift_len
        self.context_size = context_size
        self.__signal = None if signal is None else signal.copy()

    def __copy__(self):
        return YoonSpeech(signal=self.__signal, sample_rate=self.sample_rate,
                          fft_count=self.fft_count, mel_order=self.mel_order, mfcc_order=self.mfcc_order,
                          context_size=self.context_size, win_len=self.window_length, shift_len=self.shift_length)

    def get_signal(self):
        return self.__signal

    def load_sound_file(self, file_name: str):
        self.__signal, self.sample_rate = librosa.load(file_name, self.sample_rate)

    def save_sound_file(self, file_name: str):
        soundfile.write(file_name, self.__signal, self.sample_rate)

    def resampling(self, target_rate: int):
        signal = librosa.resample(self.__signal, self.sample_rate, target_rate)
        return YoonSpeech(signal=signal, sample_rate=target_rate,
                          fft_count=self.fft_count, mel_order=self.mel_order, mfcc_order=self.mfcc_order,
                          win_len=self.window_length, shift_len=self.shift_length, context_size=self.context_size)

    def crop(self, start_sec: float, end_sec: float):
        start, end = int(start_sec * self.sample_rate), int(end_sec * self.sample_rate)
        return YoonSpeech(signal=self.__signal[start:end], sample_rate=self.sample_rate,
                          fft_count=self.fft_count, mel_order=self.mel_order, mfcc_order=self.mfcc_order,
                          win_len=self.window_length, shift_len=self.shift_length, context_size=self.context_size)

    def scaling(self, dMin=-0.99999, dMax=0.99999):
        signal = minmax_scale(self.__signal, feature_range=(dMin, dMax))
        return YoonSpeech(signal=signal, sample_rate=self.sample_rate,
                          fft_count=self.fft_count, mel_order=self.mel_order, mfcc_order=self.mfcc_order,
                          win_len=self.window_length, shift_len=self.shift_length, context_size=self.context_size)

    def show_time_signal(self):
        # Init graph
        figure = matplotlib.pyplot.figure(figsize=(14, 8))
        graph = figure.add_subplot(211)
        graph.set_title('Raw Speech Signal')
        graph.set_xlabel('Time (sec)')
        graph.set_ylabel('Amplitude')
        graph.grid(True)
        # Set graph per time sample
        time_len = len(self.__signal)
        x_units = numpy.linspace(0, time_len / self.sample_rate, time_len)
        graph.set_xlim(x_units.min(), x_units.max())
        graph.set_ylim(self.__signal.min() * 1.4, self.__signal.max() * 1.4)
        graph.plot(x_units, self.__signal)
        # Show graph
        matplotlib.pyplot.show()

    @staticmethod
    def show_mfcc(feature_data: numpy.ndarray):
        # Init Graph
        matplotlib.pyplot.title('MFCC')
        matplotlib.pyplot.ylabel('The number of Coefficients')
        matplotlib.pyplot.xlabel('The number of frames')
        # Set graph per Frequency
        matplotlib.pyplot.imshow(feature_data[:, 1:].transpose(), cmap='jet', origin='lower', aspect='auto')
        matplotlib.pyplot.colorbar()
        # Show graph
        matplotlib.pyplot.show()

    @staticmethod
    def show_mel_spectrum(feature_data: numpy.ndarray):
        # Init Graph
        matplotlib.pyplot.title('Power Mel Filterbanks Spectogram')
        matplotlib.pyplot.ylabel('The number of Mel Filterbanks')
        matplotlib.pyplot.xlabel('The number of frames')
        # Set graph per Frequency
        matplotlib.pyplot.imshow(feature_data.transpose(), cmap='jet', origin='lower', aspect='auto')
        matplotlib.pyplot.colorbar()
        # Show graph
        matplotlib.pyplot.show()

    # Compute Fast Fourier Transformation
    def __fft(self):
        signal_windowed = self.__signal * numpy.hanning(len(self.__signal))
        fft_size = pow(2, int(numpy.log2(len(signal_windowed))) + 1)  # Pow of 2
        freq_data = fft(signal_windowed, fft_size)
        return freq_data

    # Compute Short Time Fourier Transformation
    def __stft(self):
        win_len = int(self.window_length * self.sample_rate)
        shift_len = int(self.shift_length * self.sample_rate)
        num_frame = int(numpy.floor((len(self.__signal) - win_len) / float(shift_len)) + 1)
        win_filter = numpy.hanning(win_len)
        sections = []
        for i in range(0, num_frame):
            sections.append(win_filter * self.__signal[i * shift_len: i * shift_len + win_len])
        frame = numpy.stack(sections)
        return numpy.fft.rfft(frame, n=self.fft_count, axis=1)

    # Combine features depending on the context window setup
    def __context_window(self, feature_data: numpy.ndarray):
        if self.context_size < 2:  # Context window length is too short
            return feature_data
        left = int(self.context_size / 2)
        right = self.context_size - left
        results = []
        for i in range(left, len(feature_data) - right):
            result = numpy.concatenate((feature_data[i - left: i + right]), axis=-1)
            results.append(result)
        return numpy.vstack(results)

    # Compute magnitude and Log-magnitude spectrum
    def get_log_magnitude(self,
                          feature_type: str = 'fft'):
        if feature_type == 'fft':
            freq_data = self.__fft()
            half_size = int(len(freq_data) / 2)  # Use only Half
            feature_data = abs(freq_data[0:half_size])
            return 20 * numpy.log10(feature_data)
        elif feature_type == 'stft':
            freq_data = self.__stft()
            feature_data = abs(freq_data)
            return 20 * numpy.log10(feature_data + 1.0e-10)
        else:
            print('Wrong Fourier transform type : {}'.format(feature_type))
            raise StopIteration

    def get_feature(self,
                    feature_type: str = 'deltas'):
        if feature_type == "org":
            return self.__signal.copy()
        # Scaling(-0.9999, 0.9999) : To protect overload error in float range
        elif feature_type == "mel":
            return self.scaling(-0.9999, 0.9999).get_log_mel_spectrum()
        elif feature_type == "mfcc":
            return self.scaling(-0.9999, 0.9999).get_mfcc()
        elif feature_type == "deltas":
            return self.get_mfcc_deltas(bContext=True)
        else:
            Exception("Feature type is not correct")

    def get_dimension(self,
                      feature_type: str = 'deltas'):
        # Dimension to original or convert dataset
        if feature_type == "org":
            return 0
        elif feature_type == "mfcc":
            return self.mfcc_order
        elif feature_type == "mel":
            return self.mel_order
        elif feature_type == "deltas":
            return self.mfcc_order * 3 * self.context_size  # MFCC * delta * delta-delta
        else:
            Exception("Feature type is not correct")

    def get_log_mel_spectrum(self):
        freq_data = self.__stft()
        feature_data = abs(freq_data) ** 2
        mel_filter = librosa.filters.mel(self.sample_rate, self.fft_count, self.mel_order)
        mel_spectrum = numpy.matmul(feature_data, mel_filter.transpose())  # Multiply Matrix
        return 10 * numpy.log10(mel_spectrum)

    def get_log_mel_deltas(self,
                           is_context: bool = False):
        # Perform a short-time Fourier Transform
        shift_len = int(self.shift_length * self.sample_rate)
        win_len = int(self.window_length * self.sample_rate)
        freq_data = librosa.core.stft(self.__signal, n_fft=self.fft_count, hop_length=shift_len, win_length=win_len)
        feature_data = abs(freq_data).transpose()
        # Estimate either log mep-spectrum
        mel_filter = librosa.filters.mel(self.sample_rate, n_fft=self.fft_count, n_mels=self.mel_order)
        power_data = feature_data ** 2
        feature_data = numpy.matmul(power_data, mel_filter.transpose())
        feature_data = 10 * numpy.log10(feature_data + numpy.array(sys.float_info.epsilon))  # feature + Epsilon
        delta1 = librosa.feature.delta(feature_data)
        delta2 = librosa.feature.delta(feature_data, order=2)
        result = numpy.concatenate(feature_data, delta1, delta2)
        if is_context:
            result = self.__context_window(result)
        return result

    def get_mfcc(self):
        log_mel_spectrum = self.get_log_mel_spectrum()
        # Discreate cosine transformation
        mfcc_data = scipy.fftpack.dct(log_mel_spectrum, axis=-1, norm='ortho')
        return mfcc_data[:, :self.mel_order]

    def get_mfcc_deltas(self,
                        bContext: bool = False):
        # Perform a short-time Fourier Transform
        shift_len = int(self.shift_length * self.sample_rate)
        win_len = int(self.window_length * self.sample_rate)
        freq_data = librosa.core.stft(self.__signal, n_fft=self.fft_count, hop_length=shift_len,
                                      win_length=win_len)
        feature_data = abs(freq_data).transpose()
        # Estimate either log mep-spectrum
        mel_filter = librosa.filters.mel(self.sample_rate, n_fft=self.fft_count, n_mels=self.mfcc_order)
        power_data = feature_data ** 2
        feature_data = numpy.matmul(power_data, mel_filter.transpose())
        feature_data = 10 * numpy.log10(feature_data + numpy.array(sys.float_info.epsilon))  # feature + Epsilon
        feature_data = scipy.fftpack.dct(feature_data, axis=-1, norm='ortho')
        delta1 = librosa.feature.delta(feature_data)
        delta2 = librosa.feature.delta(feature_data, order=2)
        result = numpy.concatenate((feature_data, delta1, delta2), axis=-1)
        if bContext:
            result = self.__context_window(result)
        return result
