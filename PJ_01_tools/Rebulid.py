import librosa
import torch
from pesq import pesq
from scipy.io import wavfile
import numpy as np
from pystoi import stoi
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality



wav = r"F:\Data\raw_data\Edinburgh Dataset\28_speakers\03_test\clean\p232_001.wav"
import scipy.signal
audio, sample_rate = librosa.load(wav, sr =16000)
#rate, audio = wavfile.read('p232_167.wav')
y_pad = librosa.util.fix_length(audio, size=len(audio) + 512 // 2, mode='constant')

spectrum,phase = librosa.magphase(librosa.stft(y_pad, window='hamming', n_fft=512, hop_length=256, win_length=512, center=True))

#spectrum2 =librosa.stft(audio, window='hann', n_fft=512, hop_length=256, win_length=512, center=False)

# get magnitude and phase from the complex numbers
spectrum = spectrum
#phase = np.unwrap(np.angle(phase))

# reconstruct real/imaginary parts from magnitude and phase
spectrum3 = spectrum * phase
#spectrum3 = spectrum *np.exp(1j*phase)
#spectrum3 = np.concatenate((np.zeros((1,spectrum3.shape[1])), spectrum3[1:256],np.zeros((1,spectrum3.shape[1]))),axis=0 )

#STFT = librosa.stft(audio, window='hann', n_fft=512, hop_length=256, win_length=512, center=False)
reconstructed_audio = librosa.istft(spectrum3, window='hamming', hop_length=256, win_length=512, center=True, length=len(audio))

wav_ref = r'C:\Users\audiolab\Desktop\rec_51.wav'

sf.write(wav_ref, reconstructed_audio, 16000, 'PCM_24')
sum(audio[:len(reconstructed_audio)] - reconstructed_audio)

rate, deg = wavfile.read("p232_001.wav")
print(pesq(rate, deg, reconstructed_audio, 'wb'))
nb_pesq = PerceptualEvaluationSpeechQuality(16000,'wb')
pesq_torch = nb_pesq(torch.from_numpy(deg), torch.from_numpy(reconstructed_audio))
print(pesq_torch.numpy())
# Clean and den should have the same length, and be 1D
print(stoi(deg,audio,sample_rate))
print(np.mean(np.abs(audio - reconstructed_audio)))

T = np.linspace(0, len(audio) / rate, num=len(audio))

fig, axs = plt.subplots(3)
fig.suptitle('Zeitbereich')
axs[0].plot(T, audio,color='b', label='raw_signal')
axs[1].plot(T, reconstructed_audio,color='r', label='reconst_signal')
axs[2].plot(T, audio-reconstructed_audio, color ='y', label ='differenz')
plt.xlabel('Time[s]')
plt.ylabel('Amplitude[dB]')
axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[0].legend()
axs[1].legend()
axs[2].legend()
#axs[2].set_ylim([-0.5, 0.5])
plt.show()

#
# # import matlab.engine
# # from scipy.io import wavfile
# # #
# # # eng = matlab.engine.start_matlab()
# wav_ref = r'C:\Users\audiolab\Desktop\rec_51.wav'
# audio, sample_rate = librosa.load(wav_ref, sr =16000)
# S = librosa.feature.melspectrogram(y=audio, sr=16000,n_fft=512,hop_length=256,n_mels=80,
#                                    fmax=8000)
# X = librosa.feature.mfcc(S=S, sr=16000, n_mfcc=13, dct_type=2, norm='ortho')
# # print(np.mean(X-X))
# # fig, ax = plt.subplots(nrows=2)
# # img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
# #                                x_axis='time', y_axis='mel', fmax=8000,
# #                                ax=ax[0])
# # fig.colorbar(img, ax=[ax[0]])
# # ax[0].set(title='Mel spectrogram')
# # ax[0].label_outer()
# # img = librosa.display.specshow(X, x_axis='time', ax=ax[1])
# # fig.colorbar(img, ax=[ax[1]])
# # ax[1].set(title='MFCC')
# # plt.show()
# import tensorflow as tf
# # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
# # wav_ref = r'C:\Users\audiolab\Desktop\rec_51.wav'
# # audio, sample_rate = librosa.load(wav_ref, sr =16000)
#
# # A 1024-point STFT with frames of 64 ms and 75% overlap.
# stfts = tf.signal.stft(audio, frame_length=512, frame_step=256,
#                        fft_length=512)
# spectrograms = tf.abs(stfts)
#
# # Warp the linear scale spectrograms into the mel-scale.
# num_spectrogram_bins = stfts.shape[-1]
# lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000.0, 13
# linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
#   num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
#   upper_edge_hertz)
# mel_spectrograms = tf.tensordot(
#   spectrograms, linear_to_mel_weight_matrix, 1)
# mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
#   linear_to_mel_weight_matrix.shape[-1:]))
#
# # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
# log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
#
# # Compute MFCCs from log_mel_spectrograms and take the first 13.
# mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
#   log_mel_spectrograms)
# print(np.mean(X-mfccs.numpy()))
#
#
# # import math
# # import numpy as np
# #
# # def sqCepDist(x, y):
# #     diff = x - y
# #     return np.inner(diff, diff)
# #
# # def eucCepDist(x, y):
# #     diff = x - y
# #     return math.sqrt(np.inner(diff, diff))
# #
# # logSpecDbConst = 10.0 / math.log(10.0) * math.sqrt(2.0)
# # def logSpecDbDist(x, y):
# #     diff = x - y
# #     return logSpecDbConst * math.sqrt(np.inner(diff, diff))
# # import math
# # def freqToMel(f):
# #     return 1127 * math.log(1 + (f/700))
# #
# # # Vectorize function to apply to numpy arrays
# # freqToMelv = np.vectorize(freqToMel)
# #
# # # Observing 0 to 10,000 Hz
# # Hz = np.linspace(0,1e4)
# # # Now we just apply the vectorized function to the Hz variable
# # Mel = freqToMelv(Hz)
# #
# # # Plotting the figure:
# # fig, ax = plt.subplots(figsize = (20,10))
# # ax.plot(Hz, Mel)
# # plt.title('Hertz to Mel')
# # plt.xlabel('Hertz Scale')
# # plt.ylabel('Mel Scale')
# # plt.show()