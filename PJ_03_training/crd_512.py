from PJ_01_tools.Data_TIMIT_Read import DataAcquisition, DataScale
import numpy as np
import pathlib
from PJ_01_tools.plot_learning import plot_learning_curves
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.callbacks as cbs
import time
import tensorflow as tf
from tensorflow.keras import backend as K
import shelve

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, Add

np.random.seed(1337)  # for reproducibility
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_physical_devices('GPU')
        print(len(gpus), 'Physical GPUs', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

mini_batch = 15
look_backward = 5
look_forward = 0
n_fft = 512
n_frames = 200
nb_epochs = 1000
batch_size = 16  # 16
learning_rate = 8e-5
w1 = 0.7
w2 = 0.3
w3 = 0
# start the training amir costumloss_w1_{w1}_w2_{w2}_w3_{w3}
result_path = pathlib.Path(
    f'F:/Projekts/SE/Phase_Korrektur/big-Timit-data/w1,w2/01_crd_model_{batch_size}_w1_{w1}_w2_{w2}_w3_{w3}_look_backward_{look_backward}_{learning_rate}_return_true_last_ReLU_my_methode_data_casual_ohne_bn')
startAll = time.time()


def si_sdr(preds, target):
    eps = tf.keras.backend.epsilon()
    preds = tf.cast(preds, dtype=tf.double)
    target = tf.cast(target, dtype=tf.double)
    alpha = (tf.math.reduce_sum(preds * target, axis=-1) + eps) / (
            tf.math.reduce_sum(target ** 2, axis=-1) + eps)

    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (tf.math.reduce_sum(target_scaled ** 2, axis=-1, keepdims=True) + eps) / (
            tf.math.reduce_sum(noise ** 2, axis=-1, keepdims=True) + eps)

    x1 = tf.math.log(tf.abs(val))
    x2 = tf.cast(tf.math.log(10.0), dtype=tf.float64)
    val_log_10 = x1 / x2

    val = 10 * val_log_10
    return val


def inverse_fft_func(fft, FFT=n_fft):
    mag = tf.reshape(fft, [fft.shape[0]* fft.shape[1],fft.shape[2]])
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


path = pathlib.Path('F:/Features/adjusted_noise_only_my_methode')
x = DataAcquisition(minibatch_size=mini_batch, repeat_faktor=126, n_frames=n_frames, n_fft=512,
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

INPUT_Noisy_SHAPE = [200, 257, 1]

input_noisy = Input(shape=INPUT_Noisy_SHAPE)
input_noise_component = Input(shape=(200,257), dtype=tf.complex64)

c0 = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same')(input_noisy)
c0_activ = tf.keras.layers.ELU()(c0)
#c0_activ_b = tf.keras.layers.BatchNormalization()(c0_activ)
c1 = tf.keras.layers.Conv2D(64, kernel_size=(2, 3), strides=(1, 2))(c0_activ)
c1_activ = tf.keras.layers.ELU()(c1)
#c1_activ_b = tf.keras.layers.BatchNormalization()(c1_activ)
c1_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c1_activ)
c2 = tf.keras.layers.Conv2D(64, kernel_size=(2, 3), strides=(1, 2))(c1_activ_zero_padded)
c2_activ = tf.keras.layers.ELU()(c2)
#c2_activ_b = tf.keras.layers.BatchNormalization()(c2_activ)
c2_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c2_activ)
c3 = tf.keras.layers.Conv2D(32, kernel_size=(2, 3), strides=(1, 2))(c2_activ_zero_padded)
c3_activ = tf.keras.layers.ELU()(c3)
#c3_activ_b = tf.keras.layers.BatchNormalization()(c3_activ)
c3_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c3_activ)
c4 = tf.keras.layers.Conv2D(32, kernel_size=(2, 3), strides=(1, 2))(c3_activ_zero_padded)
c4_activ = tf.keras.layers.ELU()(c4)
#c4_activ_b = tf.keras.layers.BatchNormalization()(c4_activ)
c4_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c4_activ)
c1_reshape = tf.keras.layers.Reshape((200, c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3]))(
    c4_activ_zero_padded)

Dense1 = tf.keras.layers.Dense(c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3],
                               activation=tf.keras.layers.ELU())(c1_reshape)
Dense1_dr = tf.keras.layers.Dropout(0.1)(Dense1)
#RNN1 = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(250, return_sequences=True))(Dense1_dr)

RNN1 = tf.compat.v1.keras.layers.CuDNNGRU(500, return_sequences=True)(Dense1_dr)
RNN1_act = tf.keras.layers.ELU()(RNN1)
RNN1_dr = tf.keras.layers.Dropout(0.1)(RNN1_act)
RNN2 = tf.compat.v1.keras.layers.CuDNNGRU(400, return_sequences=True)(RNN1_dr)
RNN2_act = tf.keras.layers.ELU()(RNN2)
RNN2_dr = tf.keras.layers.Dropout(0.1)(RNN2_act)
Dense2 = tf.keras.layers.Dense(300, activation=tf.keras.layers.ELU())(RNN2_dr)
Dense2_dr = tf.keras.layers.Dropout(0.1)(Dense2)
last_dense = tf.keras.layers.Dense(257, activation=tf.keras.layers.ELU())(Dense2_dr)

# last_dense = tf.keras.layers.Reshape((last_dense.shape[2], last_dense.shape[1]))(last_dense)
#
# Output_Mask = tf.keras.layers.Dense(1, activation=tf.keras.layers.ReLU())(last_dense)
#Output_Mask = tf.keras.layers.Reshape([Output_Mask.shape[1]*Output_Mask.shape[2]])(Output_Mask)
# calculating the magnitude
Mask = tf.cast(last_dense, tf.complex64)
Output_magnitude = Multiply()([Mask, input_noise_component])
# Output_magnitude = tf.abs(Output_magnitude)
model = Model(inputs=[input_noisy, input_noise_component], outputs=[Output_magnitude])
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
# model = load_model(model_path)

model.compile(optimizer=adam, loss=[custom_loss], metrics=['msle'], run_eagerly=True)
model.summary()

# Reduce learning rate when stop improving lr = lr*factor
reduce_LR = cbs.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=8, verbose=1, mode='auto')

# Stop training after 1000 epoches if the vali_loss not decreasing
stop_str = cbs.EarlyStopping(monitor='val_loss', patience=24, verbose=1, mode='auto')

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

tf.keras.utils.plot_model(model, to_file=result_path.joinpath('Flussdiagramm.png'), dpi=100)

# string:DDDD(.name)
ff = shelve.open(str(result_path.joinpath('history.slv')))
ff['train_loss'] = history.history['loss']
ff['val_loss'] = history.history['val_loss']
ff.close()
plot_learning_curves(history.history['loss'], history.history['val_loss'], result_path)
plt.clf()
print("> All Trainings Completed, Duration : ", time.time() - startAll)
