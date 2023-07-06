import pathlib
from enum import Enum
import tensorflow as tf
import numpy as np
from PJ_04_evaluation.pred_rec import PredictReconstruct, DataScale, ModelType, BigData
from PJ_04_evaluation.score_masked_base_all import Score
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


class Mode(Enum):
    All = 1
    Predict = 2
    Score = 3


mode = Mode.Score

model_result = pathlib.Path("E:/11.06/06_my_model_0.0001_w1_1_w2_0.8_w3_1_g2_0_old_Sdr+new_sir_omega_True_big_t11_+_t21_-_t3_compact_normlized_powernorm_divided")

look_backward = 5
mini_batch = 800
repeat_factor = 120
n_frames = 200
n_fft = 512


test_path = pathlib.Path(f"F:/test/adjusted_noise_only/test/slv_512/noisy")
ref_path_data_set = pathlib.Path("F:/test/adjusted_noise_only/test/scores")


if mode == Mode.Predict or mode == Mode.All:
    pred_rec = PredictReconstruct(FFT=n_fft,frames=n_frames, minibatch=mini_batch,
                                  input_type=DataScale.not_logarithmic,
                                  model_type=ModelType.STFT_iSTFT, big_data=BigData.big_dataset,
                                  look_backward=look_backward)
    pred_rec.test_prediction(model_path=model_result, test_data_path=test_path)

if mode == Mode.Score or mode == Mode.All:
    Score(result_path=model_result,ref_path=ref_path_data_set, repeat_faktor=repeat_factor, mode=0, dns_mos=True)





