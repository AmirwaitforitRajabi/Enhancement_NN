from big_data_prepration import DataPrepration, Mode, BigData
import pathlib
data_path = pathlib.Path("F:/Data/UNi_Gent/IEEE/data")
destination = pathlib.Path("F:/Data/UNi_Gent/IEEE/data/03_test/feat_noisy")

x = DataPrepration(FFT=512, data_set=BigData.big_dataset, input_type=Mode.test_noisy)
y = x.data_acquisition(data_path=data_path, destination=destination)
