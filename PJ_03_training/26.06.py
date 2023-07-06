from PJ_01_tools.Data_TIMIT_Read import DataAcquisition, DataScale
import numpy as np
import pathlib
from PJ_01_tools.plot_learning import plot_learning_curves
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow.python.keras.callbacks as cbs
import time
import tensorflow as tf
from tensorflow.python.keras import backend as K
import shelve

import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Multiply, Add

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
nb_epochs = 200
batch_size = 16  # 16
learning_rate = 1e-4
w1 = 1
w2 = 0.8
w3 = 1
g2 = 0
oomega = True
if w3 == 0:
    t1,t2,t3,t4 =0,0,0,0
else:
    #find the dynamic range '(1,0,0)'-(0,1,0)-(0,0,1)-(1,1,0)-(0,1,1)-(1,1,1)
    t1 =1#0.05#
    t2 =1#1#
    t3 =1#0.5#0.5

result_path = pathlib.Path(
    f'E:/11.06/06_my_model_{learning_rate}_w1_{w1}_w2_{w2}_w3_{w3}_g2_{g2}_old_Sdr+new_sir_omega_{oomega}_big_t1{t1}_+_t2{t2}_-_t3_compact_normlized_powernorm_divided')
startAll = time.time()
path = pathlib.Path('F:/Features/adjusted_noise_only_my_methode')
def weighted_signal_distortion_ratio_loss_paper_1(target, predict):

    eps = tf.keras.backend.epsilon()
    predict = tf.cast(predict, dtype=tf.float64)
    target = tf.cast(target, dtype=tf.float64)

    predict_norm = tf.norm(predict, axis=-1, ord='euclidean')
    return  (tf.math.pow(predict_norm, 2)  + eps)/tf.math.pow(tf.tensordot(target, predict,axes=1), 2)

def powernorm(x,y):
    power_x = K.sqrt(K.sum(K.abs(x) ** 2))
    power_y = K.sqrt(K.sum(K.abs(y) ** 2))
    coff = 1 / (power_x * power_y)*K.sum(K.abs(x)*K.abs(y))
    return coff






def sdt_mean(target, predict):
    eps = tf.keras.backend.epsilon()
    predict = tf.cast(predict, dtype=tf.float64)
    target = tf.cast(target, dtype=tf.float64)

    target_norm = tf.norm(target, axis=-1, ord='euclidean')
    predict_norm = tf.norm(predict, axis=-1, ord='euclidean')
    x = tf.tensordot(target, predict, axes=0) / (target_norm * predict_norm + eps)
    print(x)
    return x

def weighted_signal_distortion_ratio_loss(target, predict):

    eps = tf.keras.backend.epsilon()
    predict = tf.cast(predict, dtype=tf.float64)
    target = tf.cast(target, dtype=tf.float64)
    noise = predict - target
    noise_norm = tf.norm(noise, axis=-1, ord='euclidean')
    target_norm = tf.norm(target, axis=-1, ord='euclidean')
    predict_norm = tf.norm(predict, axis=-1, ord='euclidean')
    print(target_norm,predict_norm)

    #theta = tf.tensordot(target, noise, axes=1) / (target_norm*noise_norm+eps)
    alpha = tf.tensordot(target, predict, axes=0) / (target_norm * predict_norm + eps)
    print(alpha)
    if predict_norm*alpha > target_norm:
        gama = 1
    else:
        gama = -1

    return t1 * (target_norm - target_norm / alpha)
    #t3 * tf.abs(tf.math.pow(tf.tensordot(predict, noise, axes=1), 2) / tf.math.pow(tf.tensordot(target, predict, axes=1), 2))

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

    norm_correlation = powernorm(y_true_d_type,y_predict_d_type)
    #print(norm_correlation)
    return (w1 * K.mean(tf.abs(tf.abs(y_true_d_type) - tf.abs(y_predict_d_type)), axis=-1) + w2 * K.mean(
        tf.abs(y_true_d_type - y_predict_d_type), axis=-1)) - w3*K.mean(norm_correlation)

def custom_loss_2(y_true, y_predict):

    rec_true, y_true_d_type = inverse_fft_func(y_true)
    rec_predict, y_predict_d_type = inverse_fft_func(y_predict)
    return w3 * K.mean(K.abs(weighted_signal_distortion_ratio_loss(rec_true,rec_predict)))
####################





np.random.seed(1337)  # for reproducibility

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
input_noise_component = Input(shape=(200,257), dtype=tf.complex64)
c0 = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same')(input_noisy)
c0_activ = tf.keras.layers.ELU()(c0)
c0_activ_b = tf.keras.layers.BatchNormalization()(c0_activ)
c1 = tf.keras.layers.Conv2D(64, kernel_size=(2, 3), strides=(1, 2))(c0_activ_b)
c1_activ = tf.keras.layers.ELU()(c1)
c1_activ_b = tf.keras.layers.BatchNormalization()(c1_activ)
c1_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c1_activ_b)
c2 = tf.keras.layers.Conv2D(32, kernel_size=(2, 3), strides=(1, 2))(c1_activ_zero_padded)
c2_activ = tf.keras.layers.ELU()(c2)
c2_activ_b = tf.keras.layers.BatchNormalization()(c2_activ)
c2_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c2_activ_b)
c3 = tf.keras.layers.Conv2D(32, kernel_size=(2, 3), strides=(1, 2))(c2_activ_zero_padded)
c3_activ = tf.keras.layers.ELU()(c3)
c3_activ_b = tf.keras.layers.BatchNormalization()(c3_activ)
c3_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c3_activ_b)
c4 = tf.keras.layers.Conv2D(32, kernel_size=(2, 3), strides=(1, 2))(c3_activ_zero_padded)
c4_activ = tf.keras.layers.ELU()(c4)
c4_activ_b = tf.keras.layers.BatchNormalization()(c4_activ)
c4_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c4_activ_b)
c1_reshape = tf.keras.layers.Reshape((n_frames, c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3]))(
    c4_activ_zero_padded)

Dense1 = tf.keras.layers.Dense(c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3],
                               activation=tf.keras.layers.ELU())(c1_reshape)
Dense1_dr = tf.keras.layers.Dropout(0.1)(Dense1)
#c_skip = tf.keras.layers.Conv1D(c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3], kernel_size=1, padding='same')(Dense1_dr)

R0 = tf.compat.v1.keras.layers.CuDNNGRU(120, return_sequences=True)(Dense1_dr[:, :, :240])
R0_activ = tf.keras.layers.ELU()(R0)
R1 = tf.compat.v1.keras.layers.CuDNNGRU(120, return_sequences=True)(Dense1_dr[:, :, 240:])
R1_activ = tf.keras.layers.ELU()(R1)
R02 = tf.compat.v1.keras.layers.CuDNNGRU(120, return_sequences=True)(R0_activ)
R02_activ = tf.keras.layers.ELU()(R02)
R12 = tf.compat.v1.keras.layers.CuDNNGRU(120, return_sequences=True)(R1_activ)
R12_activ = tf.keras.layers.ELU()(R12)
R_output = tf.concat([R02_activ, R12_activ], 2)
#R_concatenating = tf.compat.v1.keras.layers.CuDNNLSTM(480, return_sequences=True)(Dense1_dr)
# R_concatenating_active = tf.keras.layers.ELU()(R_concatenating)

Dense2 = tf.keras.layers.Dense(480, activation=tf.keras.layers.ELU())(R_output)
Dense2_dr = tf.keras.layers.Dropout(0.1)(Dense2)
# last_dense_0 = tf.keras.layers.Dense(161)(Dense2_dr)
# last_dense = tf.keras.activations.sigmoid(last_dense_0)
last_dense = tf.keras.layers.Dense(257, activation=tf.keras.layers.ELU())(Dense2_dr)

# calculating the magnitude
Mask = tf.cast(last_dense, tf.complex64)
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