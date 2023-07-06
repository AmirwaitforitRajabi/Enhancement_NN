import math
import os
import pathlib

import numpy as np

from keras.callbacks import ModelCheckpoint
import keras.callbacks as cbs
import time
import tensorflow as tf
import shelve
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Multiply, Add
from PJ_01_tools.plot_learning import plot_learning_curves
from PJ_01_tools.Data_TIMIT_Read import DataAcquisition, DataScale

np.random.seed(1337)  # for reproducibility

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


mini_batch = 15
look_backward = 5
look_forward = 0
n_fft = 320
n_frames = 200
nb_epochs = 1000
batch_size = 16  # 16
learning_rate = 8e-5
w1 = 0.7
w2 = 0.3
w3 = 0


def inverse_fft_func(fft, FFT=n_fft):
    mag = tf.reshape(fft, [fft.shape[0]*fft.shape[1],fft.shape[2]])
    rec = tf.signal.inverse_stft(mag, frame_length=FFT, frame_step=int(FFT / 2), window_fn=tf.signal.hann_window)
    mag_enh = tf.signal.stft(rec, frame_length=FFT, frame_step=int(FFT / 2), window_fn=tf.signal.hann_window)
    return rec, mag_enh

def custom_loss(y_true, y_predict):

    rec_true, y_true_d_type = inverse_fft_func(y_true)
    rec_predict, y_predict_d_type = inverse_fft_func(y_predict)
    y_true_d_type = tf.cast(y_true_d_type, dtype=tf.complex64)
    y_predict_d_type = tf.cast(y_predict_d_type, dtype=tf.complex64)
    return w1 * K.mean(tf.abs(tf.abs(y_true_d_type) - tf.abs(y_predict_d_type)), axis=-1) + w2 * K.mean(
        tf.abs(y_true_d_type - y_predict_d_type), axis=-1)

####################

# start the training amir costumloss_w1_{w1}_w2_{w2}_w3_{w3}
result_path = pathlib.Path(
    f'F:/2Projekts/microsoft/06_my_model_{batch_size}_w1_{w1}_w2_{w2}_w3_{w3}_2__{learning_rate}')
startAll = time.time()



np.random.seed(1337)  # for reproducibility
path = pathlib.Path('E:/Data/feature_data/TIMIT_320')
x = DataAcquisition(minibatch_size=mini_batch, repeat_faktor=126, n_frames=n_frames, n_fft=320,
                    look_backward=look_backward, input_type=DataScale.not_logarithmic)
# add the training und validation data to the generator
x_data_train = x.read_data(data_path=path, mode='train')
x_data_valid = x.read_data(data_path=path, mode='valid')

# calculate steps_per_epoch

num_train_examples = len(x.train_audio_paths)
steps_per_epoch = num_train_examples // mini_batch
#

num_valid_samples = len(x.valid_audio_paths)
validation_steps = num_valid_samples // mini_batch

INPUT_Noisy_SHAPE = [200, 161, 1]
###################################################
# 1.1 CNN parameters
###################################################
n1 = 16
n2 = 32
n3 = 64
n4 = 128


#####################################################################################
# 3 define model
#####################################################################################

input_noisy = Input(shape=INPUT_Noisy_SHAPE)
input_noise_component = Input(shape=(200,161), dtype=tf.complex64)
# layer 1
c1 = tf.keras.layers.Conv2D(n1, kernel_size=(2, 3), strides=(1, 2))(input_noisy)
c1_activ = tf.keras.layers.LeakyReLU()(c1)

c1_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c1_activ)
#
# skip 1
c1_b_skip = tf.keras.layers.Conv2D(n1, kernel_size=(1, 1), padding='same')(c1_b_zero_padded)
#
# layer 2
c2 = tf.keras.layers.Conv2D(n2, kernel_size=(2, 3), strides=(1, 2))(c1_b_zero_padded)
c2_activ = tf.keras.layers.LeakyReLU()(c2)

c2_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c2_activ)

# skip 2
c2_b_skip = tf.keras.layers.Conv2D(n2, kernel_size=(1, 1), padding='same')(c2_b_zero_padded)

# layer 3
c3 = tf.keras.layers.Conv2D(n3, kernel_size=(2, 3), strides=(1, 2))(c2_b_zero_padded)
c3_activ = tf.keras.layers.LeakyReLU()(c3)

c3_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c3_activ)

# skip 3
c3_b_skip = tf.keras.layers.Conv2D(n3, kernel_size=(1, 1), padding='same')(c3_b_zero_padded)

# layer 4
c4 = tf.keras.layers.Conv2D(n4, kernel_size=(2, 3), strides=(1, 2))(c3_b_zero_padded)
c4_activ = tf.keras.layers.LeakyReLU()(c4)

c4_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c4_activ)

# skip 4
c4_b_skip = tf.keras.layers.Conv2D(n4, kernel_size=(1, 1), padding='same')(c4_b_zero_padded)

# LSTM-Teil
Re_shape = tf.keras.layers.Reshape((200, 9 * 128))(c4_b_zero_padded)

R0 = tf.compat.v1.keras.layers.CuDNNGRU(288, return_sequences=True)(Re_shape[:, :, :288])
R1 = tf.compat.v1.keras.layers.CuDNNGRU(288, return_sequences=True)(Re_shape[:, :, 288:576])
R2 = tf.compat.v1.keras.layers.CuDNNGRU(288, return_sequences=True)(Re_shape[:, :, 576:864])
R3 = tf.compat.v1.keras.layers.CuDNNGRU(288, return_sequences=True)(Re_shape[:, :, 864:1152])

R_output = tf.concat([R0, R1, R2, R3], 2)

Re_Reshape = tf.keras.layers.Reshape((200, 9, 128))(R_output)


Skip2 = Add()([c4_b_skip, Re_Reshape])

CT1 = tf.keras.layers.Conv2DTranspose(n3, kernel_size=(2, 3), strides=(1, 2), padding='valid')(Skip2)
CT1_activ = tf.keras.layers.LeakyReLU()(CT1)
CT1_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT1_activ)

Skip3 = Add()([c3_b_skip, CT1_re])

CT2 = tf.keras.layers.Conv2DTranspose(n2, kernel_size=(2, 3), strides=(1, 2), padding='valid')(Skip3)
CT2_activ = tf.keras.layers.LeakyReLU()(CT2)
CT2_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT2_activ)

Skip4 = Add()([c2_b_skip, CT2_re])

CT3 = tf.keras.layers.Conv2DTranspose(n1, kernel_size=(2, 3), strides=(1, 2), padding='valid')(Skip4)
CT3_activ = tf.keras.layers.LeakyReLU()(CT3)
CT3_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT3_activ)
Zero_Padding = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 1)))(CT3_re)

Skip5 = Add()([c1_b_skip, Zero_Padding])

CT4 = tf.keras.layers.Conv2DTranspose(1, kernel_size=(2, 3), strides=(1, 2), padding='valid')(Skip5)
CT4_activ = tf.keras.activations.sigmoid(CT4)
CT4_re = tf.keras.layers.Reshape([CT4_activ.shape[1], CT4_activ.shape[2] * CT4_activ.shape[3]])(CT4_activ)
Output = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT4_re)
Mask = tf.cast(Output, tf.complex64)
Output_magnitude = Multiply()([Mask, input_noise_component])
model = Model(inputs=[input_noisy, input_noise_component], outputs=[Output_magnitude])

adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

model.compile(optimizer=adam, loss=[custom_loss], metrics=['msle'],run_eagerly=True)

model.summary()
# Reduce learning rate when stop improving lr = lr*factor
reduce_LR = cbs.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=8, verbose=1, mode='auto')

# Stop training after 1000 epoches if the vali_loss not decreasing
stop_str = cbs.EarlyStopping(monitor='val_loss', patience=16, verbose=1, mode='auto')

checkpoints_path = result_path.joinpath('Checkpoints')
if not checkpoints_path.exists():
    checkpoints_path.mkdir(parents=True, exist_ok=True)

weights_path = checkpoints_path.joinpath('resetmodel{epoch:04d}.h5')

checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

callbacks_list = [checkpoint]
history = model.fit(x.batch_train(num_train_examples), batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                    epochs=nb_epochs,
                    validation_data=x.batch_valid(num_valid_samples),
                    validation_steps=validation_steps,
                    callbacks=[callbacks_list, reduce_LR, stop_str], verbose=1, shuffle=True)
save_model_path = checkpoints_path.joinpath('resetmodel0000.h5')
model.save(save_model_path)

#tf.keras.utils.plot_model(model, to_file=result_path.joinpath('Flussdiagramm.png'), dpi=100)

# string:DDDD(.name)
ff = shelve.open(str(result_path.joinpath('history.slv')))
ff['train_loss'] = history.history['loss']
ff['val_loss'] = history.history['val_loss']
ff.close()
plot_learning_curves(history.history['loss'], history.history['val_loss'], result_path)
plt.clf()
print("> All Trainings Completed, Duration : ", time.time() - startAll)