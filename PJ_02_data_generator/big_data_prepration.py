import glob
import pickle
import shelve
import numpy as np
import os
import librosa
from enum import Enum
import time
import pathlib


# path
#####################################################################################################
# data_path = pathlib.Path("F:/Data/Edinburgh")
# destination = pathlib.Path("F:/Data/input_feature")


# init the mode of data accusation
#####################################################################################################
class Mode(Enum):
    train = 1
    valid = 2
    test_noisy = 3


class BigData(Enum):
    big_dataset = 0,
    small_dataset = 1


class DataPrepration:
    def __init__(self, FFT: int = 512, data_set: BigData = BigData.big_dataset,
                 input_type: Mode = Mode.train):

        self.FFT = FFT
        self.overlap = int(FFT / 2)
        self.input_type = input_type
        self.data_set = data_set
        self.file_names = []
        self.file_lengths = []

    def data_acquisition(self, data_path, destination):
        print('> Loading wav data... ')

        if self.input_type == Mode.test_noisy:
            noisy_test_path = data_path.joinpath('03_test', 'noisy')
            noisy_test_path_wav = list(noisy_test_path.glob('**/*.wav'))
            self._save_the_input(input_path=noisy_test_path_wav, des_path=destination, x='noisy_test')
            return

        if self.input_type == Mode.train:
            dir_name = '01_train'
            clean_name = 'clean_train'
            noisy_name = 'noisy_train'
        elif self.input_type == Mode.valid:
            dir_name = '02_valid'
            clean_name = 'clean_valid'
            noisy_name = 'noisy_valid'
        else:
            return
        clean_path = data_path.joinpath(dir_name, 'clean')
        noisy_path = data_path.joinpath(dir_name, 'noisy')
        clean_path_wav = list(clean_path.glob('**/*.wav'))
        noisy_path_wav = list(noisy_path.glob('**/*.wav'))
        if self.data_set == BigData.small_dataset:
            self._save_the_input(input_path=clean_path_wav, des_path=destination, x=clean_name)
            self._save_the_input(input_path=noisy_path_wav, des_path=destination, x=noisy_name)
        elif self.data_set == BigData.big_dataset:
            self._save_the_input(input_path=clean_path_wav, des_path=destination, x=clean_name)
            self._save_the_input(input_path=noisy_path_wav, des_path=destination, x=noisy_name)

    def _read_data(self, pathlist):
        if self.data_set == BigData.small_dataset:
            self.file_names.clear()
            sum_of_input_signal = []
            for count, line in enumerate(pathlist):
                # read dada
                sig, _ = librosa.load(line, sr=16000)
                if self.input_type == Mode.test_noisy:
                    self.file_lengths.append(len(sig))
                    self.file_names.append(line.name)
                sum_of_input_signal.append(sig)
            sum_of_input_signal_concat = np.concatenate(sum_of_input_signal, axis=0)
            self.sig_length_fix = librosa.util.fix_length(sum_of_input_signal_concat,
                                                          size=len(
                                                              sum_of_input_signal_concat) + self.overlap)

        elif self.data_set == BigData.big_dataset:
            sig, _ = librosa.load(str(pathlist), sr=16000)
            #self.sig_length_fix = librosa.util.fix_length(sig, size=len(sig) + self.overlap)
            self.sig_length_fix = sig
            self.file_lengths = len(sig)
            self.file_names = pathlist.name

    def _calculate_fft(self, path):
        self._read_data(path)
        self.complex_spec = librosa.stft(self.sig_length_fix, window='hann', n_fft=self.FFT,
                                         hop_length=self.overlap,
                                         win_length=self.FFT, center=True)
        self.mag, self.phase = librosa.magphase(
            librosa.stft(self.sig_length_fix, window='hann', n_fft=self.FFT, hop_length=self.overlap,
                         win_length=self.FFT, center=True))

    def _save_the_input(self, input_path, des_path, x):
        destination_path = des_path.joinpath(self.input_type.name, x)
        if not destination_path.exists():
            destination_path.mkdir(parents=True, exist_ok=True)

        if self.data_set == BigData.small_dataset:
            self._calculate_fft(input_path)
            g = shelve.open(str(destination_path.joinpath("input_" + x + ".slv")),
                            protocol=pickle.HIGHEST_PROTOCOL)

            g['abs_spectrogram_' + x] = self.mag.T
            g['complex_spectrogram_' + x] = self.complex_spec.T

            if self.input_type == Mode.test_noisy:
                g['phase_' + x] = self.phase.T
                g["file_length_" + x] = self.file_lengths
                g["file_name_" + x] = self.file_names
            g.close()

        if self.data_set == BigData.big_dataset:
            for i in range(len(input_path)):
                self._calculate_fft(input_path[i])
                g = shelve.open(str(destination_path.joinpath(input_path[i].name[:-4] + '_' + x + ".slv")),
                                protocol=pickle.HIGHEST_PROTOCOL)

                g['abs_spectrogram_' + x] = self.mag.T
                g['complex_spectrogram_' + x] = self.complex_spec.T

                if self.input_type == Mode.test_noisy:
                    g['phase_' + x] = self.phase.T
                    g["file_length_" + x] = self.file_lengths
                    g["frame_length_" + x] = self.mag.shape[1]
                    g["file_name_" + x] = self.file_names
                g.close()
