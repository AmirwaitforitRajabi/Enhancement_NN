
import os
import time

import torch
import pathlib

import librosa
import shelve
from itertools import repeat
import numpy as np

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

import pysepm
from pystoi import stoi
import warnings
warnings.filterwarnings('ignore')
from PJ_04_evaluation.dns_mos import main, Model_Type

clean_path_data_set = pathlib.Path("E:/Data/raw_data/Timi/03_test_small/clean")
result_enhanced_path = pathlib.Path(
    "E:/Data/feature_data/TIMIT_320/test_noisy/wav")
ref_path_data_set = pathlib.Path("E:/Data/feature_data/TIMIT_320/test_noisy/wav/scores")


# mode 1 if u want to evaluate the data set
def Score(clean_path=clean_path_data_set, result_path=result_enhanced_path, ref_path=ref_path_data_set, sr=16000,
          repeat_faktor=120, mode=1, dns_mos=True):
    clean_path = list(clean_path.glob('**/*.wav'))
    clean_path = [p for p in clean_path for p in repeat(p, repeat_faktor)]
    enhanced_path = list(result_path.joinpath('Enhanced_Audio').iterdir())
    startAll = time.time()
    scores_path = result_path.joinpath('scores')

    if not scores_path.exists():
        scores_path.mkdir(parents=True, exist_ok=True)

    for i in range(len(enhanced_path)):
        enhanced_path_wav_data = list(enhanced_path[i].glob('*.wav'))
        name, pes_torchh, stois, fwSNRsegs, LLRs, dns, si_sdrs = [], [], [], [], [], [],[]

        for j in range(len(clean_path)):
            clean_sig, _ = librosa.load(clean_path[j], sr=16000)
            enhanced_sig, _ = librosa.load(enhanced_path_wav_data[j], sr=16000)
            name.append(enhanced_path_wav_data[j].name)
            if len(clean_sig) < len(enhanced_sig):
                enhanced_sig = enhanced_sig[:len(clean_sig)]
            else:
                clean_sig = clean_sig[:len(enhanced_sig)]

            # Speech Quality Measures
            # 1 Perceptual Evaluation Speech Quality
            nb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
            pesq_torch = nb_pesq(torch.from_numpy(enhanced_sig), torch.from_numpy(clean_sig))
            pes_torchh.append(pesq_torch.numpy())

            # 2 Log likehood Ratio
            LLR = pysepm.qualityMeasures.llr(clean_sig, enhanced_sig, sr)
            LLRs.append(LLR)
            # # 3 Frequency-weighted Segmental SNR
            fwSNRseg_1 = pysepm.qualityMeasures.fwSNRseg(clean_sig, enhanced_sig, sr)
            fwSNRsegs.append(fwSNRseg_1)

            # Speech intelligibility Measures
            # 3 Short-time objective intelligibility
            st = stoi(clean_sig, enhanced_sig, sr)
            stois.append(st)

            # 4 Scale-Invariant Signal to Distortion Ratio (SI-SDR)
            si_sdr_func = ScaleInvariantSignalDistortionRatio()
            si_sdr = si_sdr_func(torch.from_numpy(enhanced_sig), torch.from_numpy(clean_sig))
            si_sdrs.append(si_sdr.numpy())
        if dns_mos:
            #5 DNS-Mos
            dns_raw = main(enhanced_path_wav_data, model_type=Model_Type.not_personalized_MOS)
            for value in dns_raw:
                dns.append(value[1])

        else:
            dns = np.zeros(shape=len(enhanced_path_wav_data))

        print(str(enhanced_path[i].name))
        Q = 2 * np.mean(dns) + np.mean(pes_torchh) - np.mean(LLRs) + 0.3 * np.mean(fwSNRsegs) + 0.2 * np.mean(
            si_sdrs)

        print('     PESQ_torch: %s' % np.mean(pes_torchh))
        print('     fwSNRseg: %s' % np.mean(fwSNRsegs))
        print('     LLR: %s' % np.mean(LLRs))
        print('     STOI: %s' % np.mean(stois))
        print('     si_sdr: %s' % np.mean(si_sdrs))
        print('     DNS: %s' % np.mean(dns))

        print('     Quality of Model: %s' % np.mean(Q))
        print("> All Tests Completed, Duration : ", time.time() - startAll)

        g = shelve.open(str(scores_path.joinpath('score' + enhanced_path[i].name[20:] + '.slv')))
        g['name'] = name
        g['pesq_torch'] = pes_torchh
        g['LLR'] = LLRs
        g['fwSNRseg'] = fwSNRsegs
        g['STOI'] = stois
        g['si_sdr'] = si_sdrs
        g['dns'] = dns
        g['Quality'] = Q
        g.close()

        if mode == 0:
            # there is a ref.score of noisy dataset
            h = shelve.open(os.path.join(ref_path, 'score_0000.slv'))
            pesq_ed = np.array(h['pesq_torch'])
            LLR_ed = np.array(h['LLR'])
            STOI_ed = np.array(h['STOI'])
            fwSNRseg_ed = np.array(h['fwSNRseg'])
            si_sdr_ed = np.array(h['si_sdr'])
            dns_ed = np.array(h['dns'])
            Q_ed = np.array(h['Quality'])

            h.close()
            print('__________________******************************__________________')

            a = np.array(pes_torchh) - pesq_ed
            b = np.array(fwSNRsegs) - fwSNRseg_ed
            c = np.array(LLRs) - LLR_ed
            d = np.array(stois) - STOI_ed
            e = np.array(si_sdrs) - si_sdr_ed
            f = np.array(dns) - dns_ed
            o = Q - Q_ed

            print('     Delta PESQ_torch %s' % np.mean(a))
            print('     Delta fwSNRseg %s' % np.mean(b))
            print('     Delta LLR  %s' % np.mean(c))
            print('     Delta Stoi %s' % np.mean(d))
            print('     Delta si_sdr %s' % np.mean(e))
            print('     Delta MOS %s' % np.mean(f))
            print('     Delta Quality %s' % np.mean(o))
            print('__________________##############################__________________')

            g = shelve.open(str(scores_path.joinpath('score' + enhanced_path[i].name[20:] + '.slv')))
            g['delta_pesq_torch'] = a
            g['delta_fwSNRseg'] = b
            g['delta_LLR'] = c
            g['delta_STOI'] = d
            g['delta_si_sdr'] = e
            g['delta_mos'] = f
            g['delta_quality'] = o
            g.close()
    print("> All Tests Completed, Duration : ", time.time() - startAll)


if __name__ == '__main__':
    Score()
