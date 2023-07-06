import pathlib
import time

import librosa

from PJ_02_data_generator.generate_data_set import DataAcquisition, MixMethode, DataAugmentation,Mode

clean_path = pathlib.Path('E:/Data/raw_data/Timi/01_Train/clean')
noise = pathlib.Path('E:/Data/raw_data/Timi/05_tools/Noisy_Type/test')
destination = pathlib.Path('F:/Features/adjusted_noise_only_my_methode/test_noisy')

start = time.time()
X = DataAcquisition(fft=320, clean_speech_folder=clean_path, noise_folder=noise, SNrs=[-5, 0, 5, 10, 15, 20],
                    destination=destination, gen_mode=MixMethode.gain_clean_methode, aug_mode=DataAugmentation.no_filter,
                    NoiseInit=True, input_type=Mode.test_noisy)
X.acquire_data()
ende = time.time()
del X

print("> Saving Completed, Time : ", ende - start)
print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
