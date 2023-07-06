import pathlib
import shelve
import os
import matplotlib.pyplot as plt
import pathlib
from IPython.display import display, clear_output
from numpy import linspace
from PJ01_tools.plot_learning import plot_learning_curves
import numpy as np
import glob
# basepath = pathlib.Path("F:/Projekts/Cruze/Results/1_New_Aproach/FFT=512/Phase_Korrektur/Ent_to_end_in_time/log_power_input/1_term_loss/Timit/02_einzeln_without_1_0.0003_crd_MASKED_complex_test_15_si_sdr")
basepath = r"F:\Data\raw_data\Timi\03_test\scores"
ref_path =  r'F:\Data\UNi_Gent\IEEE\data\scores'
pathlist1 = [p for p in glob.glob(basepath[:-1] + "**/*.slv.dat", recursive=True)]
mos =True
for i in range(len(pathlist1)):
    g = shelve.open(pathlist1[i][:-4])
    klist = list(g.keys())
    print(klist)
    pes = np.array(g['pesq_torch'])
    stoi = np.array(g['STOI'])
    LLRs = np.array(g['LLR'])
    fwSNRsegs = np.array(g['fwSNRseg'])
    si_sdrs = np.array(g['si_sdr'])
    delta_pes = g['delta_pesq_torch']
    delta_fwSNRseg = g['delta_fwSNRseg']
    delta_LLR = g['delta_LLR']
    delta_stoi = g['delta_STOI']
    delta_si_sdr = g['delta_si_sdr']
    delta_qaulity = g['delta_quality']
    Q = np.array(g['Quality'])
    if mos:
        dns = np.array(g['dns'])
        delta_mos = g['delta_mos']
        print('     DNS_MOS: %s' % np.mean(dns))
        print('     DELTA_ DNS_MOS: %s' % np.mean(delta_mos))
    g.close()
    print('The Models name is:%s'% str(os.path.basename(pathlist1[i][:-8])))
    print('     PESQ: %s'% np.mean(pes))
    print('     STOI: %s'% np.mean(stoi))
    print('     LLR: %s' % np.mean(LLRs))
    print('     fwSNRseg: %s' % np.mean(fwSNRsegs))
    print('     si_sdrs: %s' % np.mean(si_sdrs))
    print('     Quality of Model: %s' % np.mean(Q))
    print('__________________******************************__________________')

    print('     Delta PESQ %s'% np.mean(delta_pes))
    print('     Delta STOI %s'% np.mean(delta_stoi))

    # if np.mean(delta_LLR) > 0:
    #     print('LLR_Positve')
    print('     Delta SNRseg %s' % np.mean(delta_LLR))
    print('     Delta fwSNRseg %s' % np.mean(delta_fwSNRseg))
    print('     Delta si_sdr %s' % np.mean(delta_si_sdr))
    print('     Delta si_sdr %s' % np.mean(delta_si_sdr))
    print('     Delta delta_qaulity %s' % np.mean(delta_qaulity))
    print('__________________##############################__________________')


#
# t = linspace(0,4, len(pesq))
# plt.plot(t,pesq, '.b')
# plt.plot(t,SNRseg, '.r')
# plt.plot(t,stois, '.k')
# plt.plot(t,fwSNRseg, '.m')
# #plt.show()
# ff = shelve.open(str(basepath.joinpath('history.slv')))
# klist = list(ff.keys())
# print(klist)
# train = ff['train_loss']
# val = ff['val_loss']
# plot_learning_curves(train, val, basepath)
# plt.show()
# print(train)
# ff.close()
