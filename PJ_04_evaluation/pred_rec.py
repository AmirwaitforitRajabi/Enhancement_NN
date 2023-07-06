import shelve
import enum
import os
from itertools import repeat

import tensorflow as tf
import time
import soundfile as sf
import librosa
import numpy as np
from sklearn import preprocessing
import tensorflow.python.keras.backend as backend
from keras.models import load_model
import scipy.io as sio
import glob

###################################################
# 0 Load data
###################################################
np.random.seed(1337)  # for reproducibility


class TimeStorage:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time


class TimeController:
    def __init__(self, read, predict):
        self.read: TimeStorage = read
        self.predict: TimeStorage = predict

    def __str__(self):
        output = "Read"
        formatted_duration = '{:5.3f}s'.format(self.read.duration)
        output = f"{output}\n{formatted_duration}\nPredict"
        formatted_duration = '{:5.3f}s'.format(self.predict.duration)
        output = f"{output}\n{formatted_duration}"
        return output


class DataScale(enum.Enum):
    logarithmic = 0,
    normalized_logarithmic = 1,
    not_logarithmic = 2


class ModelType(enum.Enum):
    magnitude = 0,
    masked_base = 1,
    in_time = 2,
    end_to_end = 3,
    STFT_iSTFT = 4

class BigData(enum.Enum):
    big_dataset = 0,
    small_dataset = 1


class ShelveFile:
    __slots__ = ["file_lengths", "complex_spectrogram_test_noisy", "abs_spectrogram_noisy", "file_names", "phase_noisy", "length", "shapes_of_heart"]

    def __init__(self, file_lengths, complex_spectrogram_test_noisy, abs_spectrogram_noisy, file_names, phase_noisy, shapes):
        self.file_lengths = file_lengths
        self.complex_spectrogram_test_noisy = complex_spectrogram_test_noisy
        self.abs_spectrogram_noisy = abs_spectrogram_noisy
        self.file_names = file_names
        self.phase_noisy = phase_noisy
        self.shapes_of_heart = shapes
        self.length = np.sum(self.file_lengths)


class PredictReconstruct:
    def __init__(self, FFT: int = 512, frames: int = 20, minibatch: int = 15, look_backward: int = 2,
                 input_type: DataScale = DataScale.logarithmic,
                 model_type: ModelType = ModelType.magnitude,
                 big_data: BigData = BigData.big_dataset):
        self.noisy_path = None
        self.FFT = FFT
        self.test_index = 0
        self.look_backward = look_backward
        self.big_data = big_data
        self.overlap = int(self.FFT / 2) + 1
        self.frames = frames
        self.minibatch = minibatch
        self.model_type = model_type
        self.input_type = input_type
        self.current_shelve_file: ShelveFile = None

    def __del__(self):
        print('class is deleted!!!')


    # def read_data(self, test_data_path, file_name='input_noisy_test.slv'):
    #     print('  >> Loading input noisy test data... ')
    #     if self.big_data == BigData.big_dataset:
    #         noisy_path_pure = test_data_path.joinpath('noisy_test')
    #         self.noisy_path = list(noisy_path_pure.glob("**/*.slv.dat"))
    #
    #
    #     elif self.big_data == BigData.small_dataset:
    #         input_test = shelve.open(os.path.join(test_data_path, file_name))
    #         complex_spectrogram_test_noisy = input_test['complex_spectrogram_noisy_test']
    #         file_lengths = input_test['file_length_noisy_test']
    #         abs_spectrogram_noisy = input_test['abs_spectrogram_noisy_test']
    #         file_names = input_test['file_name_noisy_test']
    #         phase_noisy = input_test["phase_noisy_test"]
    #         input_test.close()
    #     else:
    #         return print('no valid data available')

    def _read_data_indicis(self, test_path):
        abs_spectrogram_noisy = []
        complex_spectrogram_noisy = []
        file_lengths = []
        file_names = []
        phase_noisy = []
        shapes = []
        for path in test_path:
            # loose .dat from path string and open shelve
            data_input = shelve.open(str(path)[:-4])
            # print(list(data_input.keys()))
            abs = data_input['abs_spectrogram_noisy']
            mag = data_input['complex_spectrogram_noisy']
            file_length = data_input['file_length_noisy']
            file_name = data_input['file_name_noisy']
            phases = data_input['phase_noisy']
            shape = data_input['frame_length_noisy']
            data_input.close()
            file_lengths.append(file_length)
            phase_noisy.append(phases)
            shapes.append(shape)
            abs_spectrogram_noisy.append(abs)
            complex_spectrogram_noisy.append(mag)
            file_names.append(file_name)
        abs_spectrogram_noisy = np.concatenate(abs_spectrogram_noisy, axis=0)
        complex_spectrogram_test_noisy = np.concatenate(complex_spectrogram_noisy, axis=0)
        phase_noisy = np.concatenate(phase_noisy, axis=0)
        self.lengths = np.sum(file_lengths)
        self.current_shelve_file = ShelveFile(file_lengths, complex_spectrogram_test_noisy, abs_spectrogram_noisy,
                                              file_names,
                                              phase_noisy,shapes)

    def test_prediction(self, model_path,test_data_path):
        print('> Loading Models... ')
        model_paths = [p for p in glob.glob(os.path.join(model_path, 'Checkpoints', "**/*.h5"), recursive=True)]
        #noisy_path_pure = test_data_path.joinpath('noisy_test')
        noisy_path = list(test_data_path.glob("**/*.slv.dat"))
        for i in range(len(model_paths)):
            model = load_model(model_paths[i], compile=False)
            if i == 0:
                model.summary()
            print('     The predicited Model: %s' % str(os.path.basename(model_paths[i])))

            for k in range(int(len(noisy_path)/self.minibatch)+1):
                if self.test_index + self.minibatch < len(noisy_path):
                    path_small = [a for a in noisy_path[self.test_index:self.test_index + self.minibatch]]
                else:
                    path_small = [a for a in noisy_path[self.test_index - self.minibatch:]]
                self._read_data_indicis(test_path=path_small)

                if self.model_type == ModelType.masked_base:
                    results_on_input_noisy = model.predict([self.input_test_noisy_re, self.input_test_noisy], batch_size=16)

                # elif self.model_type == ModelType.end_to_end:
                #     results_on_input_noisy = model.predict(self.input_test_noisy_re, batch_size=16)
                #
                #
                # elif self.model_type == ModelType.magnitude:
                #     phase_padded, _ = self.reshape(self.current_shelve_file.phase_input, self.frames, self.FFT)
                #
                #     [results_on_input_noisy, self.results_on_input_noisy_magnitude] = model.predict(
                #         [self.input_test_noisy_re, self.input_test_noisy, phase_padded], batch_size=16)
                #
                # elif self.model_type == ModelType.in_time:
                #     phase_padded, _ = self.reshape(self.current_shelve_file.phase_input, self.frames, self.FFT)
                #
                #     [results_on_input_noisy, _] = model.predict([self.input_test_noisy_re, phase_padded], batch_size=16)

                elif self.model_type == ModelType.STFT_iSTFT:
                    if self.input_type == DataScale.logarithmic:
                        test_input_noisy_abs = np.log10(self.current_shelve_file.abs_spectrogram_noisy ** 2)
                    elif self.input_type == DataScale.normalized_logarithmic:
                        scaler = preprocessing.StandardScaler()
                        test_input_noisy_abs = scaler.fit_transform(
                            np.log10(self.current_shelve_file.abs_spectrogram_noisy ** 2))
                    elif self.input_type == DataScale.not_logarithmic:
                        test_input_noisy_abs = self.current_shelve_file.abs_spectrogram_noisy
                    else:
                        return print('test_input_noisy_abs is not exist!!!!')
                    #
                    test_input_noisy_abs_reshaped, _ = self._input_reshape(test_input_noisy_abs)
                    _, input_test_noisy_complex_reshaped = self._input_reshape(
                        self.current_shelve_file.complex_spectrogram_test_noisy)

                    # test_input_noisy_abs_reshaped = self._reshape_backward(test_input_noisy_abs,
                    #                                                        look_backward=self.look_backward,
                    #                                                        look_forward=0,
                    #                                                        complex_value=False)
                    # input_test_noisy_complex_reshaped = np.reshape(self.current_shelve_file.complex_spectrogram_test_noisy,(-1,257,1))
                    results_on_input_noisy = model.predict(
                        [test_input_noisy_abs_reshaped,input_test_noisy_complex_reshaped], batch_size=16)


                results_on_input_noisy = results_on_input_noisy.reshape(
                    results_on_input_noisy.shape[0]*results_on_input_noisy.shape[1],results_on_input_noisy.shape[2])
                reconstructed_enhanced_audio = self._calc_enhanced_spec(nn_output=results_on_input_noisy)
                # tf.keras.utils.plot_model(model, to_file=model_path.joinpath('Flussdiagramm.png'), dpi=500,
                #                           show_shapes=True)
                self._create_wav_files(reconstructed_enhanced_audio, model_path, model_paths, i)
                self.test_index += self.minibatch
            if self.test_index >= len(noisy_path) - self.minibatch:
                self.test_index = 0

    def _create_wav_files(self, reconstructed_enhanced_audio, model_path, model_paths, i):
        current_file_position = 0
        for j in range(len(self.current_shelve_file.file_lengths)):

            temp_data = reconstructed_enhanced_audio[
                        current_file_position:(current_file_position + self.current_shelve_file.shapes_of_heart[j])]
            current_file_position = current_file_position + self.current_shelve_file.shapes_of_heart[j]
            rec = librosa.istft(temp_data.T, window='hann', hop_length=int(self.FFT / 2), win_length=self.FFT, center=True,
                                length=self.current_shelve_file.file_lengths[j])
            destination_2 = os.path.join(model_path, 'Enhanced_Audio',
                                         'final_enhanced_audio' + '_' + os.path.basename(model_paths[i])[10:14])
            if not os.path.exists(destination_2):
                os.makedirs(destination_2)

            sf.write(os.path.join(destination_2, self.current_shelve_file.file_names[j]), rec, 16000, 'PCM_24')
            backend.clear_session()

    def _input_reshape(self, data):
        wert = data.shape[0] % self.frames
        if wert > 0:
            data = np.append(data, np.zeros([self.frames - wert, self.overlap]), axis=0)
        data_3d = data.reshape((int(data.shape[0] / self.frames), self.frames, self.overlap))
        data_4d = data_3d.reshape([-1, data_3d.shape[1], data_3d.shape[2], 1])
        return data_4d, data_3d

    def _reshape_backward(self, data, look_backward, look_forward, complex_value=False, _4D=True):

        new_dim_len = look_forward + look_backward + 1
        if complex_value:
            data_matrix_out = np.zeros((data.shape[0], new_dim_len, 257), dtype=np.complex)
        else:
            data_matrix_out = np.zeros((data.shape[0], new_dim_len, 257))
        for i in range(data.shape[0]):
            for j in range(-look_backward, look_forward + 1):
                if i < look_backward:
                    idx = max(i + j, 0)
                elif i >= data.shape[0] - look_forward:
                    idx = min(i + j, data.shape[0] - 1)
                else:
                    idx = i + j
                data_matrix_out[i, j + look_backward] = data[idx, :]
        if _4D:
            return data_matrix_out.reshape([-1, data_matrix_out.shape[1], data_matrix_out.shape[2], 1])
        else:
            return data_matrix_out

    def _calc_enhanced_spec(self, nn_output):
        # wert = self.current_shelve_file.abs_spectrogram_noisy.shape[0] % self.frames
        # if wert > 0:
        #     diff = self.frames - wert
        #     nn_output = nn_output[0:-diff]
        # else:
        #     nn_output = nn_output

        if self.model_type == ModelType.STFT_iSTFT:
            #why is there a diffrance in quality?????????????????????
            spectrum = nn_output * self.current_shelve_file.phase_noisy
            #spectrum = np.abs(nn_output) * self.current_shelve_file.phase_noisy
        else:
            spectrum = nn_output
        return spectrum
