import shelve
import os
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output
from numpy import linspace
from PJ_01_tools.plot_learning import plot_learning_curves
import numpy as np
import glob
from PJ_01_tools.plot_bar import plot_bar, read_snr_data

basepath_model1 = r'C:\Users\audiolab\Desktop\results\CRUSE\cruse320-2xLSTM1-concate skip\paper'
basepath_model2_1 = r'C:\Users\audiolab\Desktop\results\CRUSE\cruse320-2xLSTM2-concate skip\paper 07.06'
basepath_model2_03 =  r'C:\Users\audiolab\Desktop\results\CRUSE\cruse320-2xLSTM2-add skip'
Microsft = r'C:\Users\audiolab\Desktop\results\microsoft\scores'
CRD_512 = r'C:\Users\audiolab\Desktop\results\crd512\paper 07.06'
CRD_320_small = r'C:\Users\audiolab\Desktop\results\crd\crd-small\scores'
CRD_320_opt = r"C:\Users\audiolab\Desktop\results\crd\CRD_320-2xGRU2\scores"
#basepath = r'E:\Projekts\Speech_Enhancement\01_New_Approach\FFT=512\Phase_Korrektur\Timit\01_crd_model_2_inp1_amp_inp2_mag_15_costumloss_w1_0.7_w2_0_w3_0.3\scores\\'
# pathlist1 = [p for p in glob.glob(basepath[:-1] + "**/*.slv.dat", recursive=True)]

basepath3 = r'F:\test\adjusted_noise_only\test\scores'

# b = shelve.open(os.path.join(basepath3,'score_0000.slv'))
# dns = np.array(b['dns'])
# stoi = np.array(b['STOI'])
# pesq = np.array(b['pesq_torch'])
# #si_sdr = np.array(b['si_sdr'])
# b.close()
# dns_sep = read_snr_data(dns)
# pesq_sep = read_snr_data(pesq)
# stoi_sep = read_snr_data(stoi)
# #si_sdr_sep = read_snr_data(si_sdr)
#
#
# c = shelve.open(os.path.join(basepath_model1,'score_0020.slv'))
#
# dns_1= np.array(c['dns'])
# stoi_1= np.array(c['STOI'])
# pesq_1 = np.array(c['pesq_torch'])
# #si_sdr_1 = np.array(c['si_sdr'])
# b.close()
# dns_sep_1 = read_snr_data(dns_1)
# pesq_sep_1 = read_snr_data(pesq_1)
# stoi_sep_1 = read_snr_data(stoi_1)
# #si_sdr_sep_1 = read_snr_data(si_sdr_1)
#
#
# c = shelve.open(os.path.join(basepath_model2_1,'score_0038.slv'))
# dns_2 = np.array(c['dns'])
# stoi_2 = np.array(c['STOI'])
# pesq_2 = np.array(c['pesq_torch'])
# #si_sdr_1 = np.array(c['si_sdr'])
# b.close()
# dns_sep_model2_1 = read_snr_data(dns_2)
# pesq_sep_model2_1 = read_snr_data(pesq_2)
# stoi_sep_model2_1 = read_snr_data(stoi_2)
# #si_sdr_sep_1 = read_snr_data(si_sdr_1)
#
#
# c = shelve.open(os.path.join(basepath_model2_03,'score_0015.slv'))
# name = np.array(c['name'])
# dns_3 = np.array(c['dns'])
# stoi_3 = np.array(c['STOI'])
# pesq_3 = np.array(c['pesq_torch'])
# #si_sdr_1 = np.array(c['si_sdr'])
# b.close()
# dns_sep_03 = read_snr_data(dns_3)
#
# pesq_sep_03 = read_snr_data(pesq_3)
# stoi_sep_03 = read_snr_data(stoi_3)
#
#
# c = shelve.open(os.path.join(Microsft,'score_0016.slv'))
# dns_4 = np.array(c['dns'])
# stoi_4 = np.array(c['STOI'])
# pesq_4 = np.array(c['pesq_torch'])
# si_sdr_1 = np.array(c['si_sdr'])
# b.close()
# si_sdr_sep_1 = read_snr_data(si_sdr_1)
# dns_sep_04 = read_snr_data(dns_4)
# pesq_sep_04 = read_snr_data(pesq_4)
# stoi_sep_04 = read_snr_data(stoi_4)
#
# c = shelve.open(os.path.join(CRD_512,'score_0027.slv'))
# dns_5 = np.array(c['dns'])
# stoi_5 = np.array(c['STOI'])
# pesq_5 = np.array(c['pesq_torch'])
# si_sdr_1 = np.array(c['si_sdr'])
# b.close()
# dns_sep_05 = read_snr_data(dns_5)
# pesq_sep_05 = read_snr_data(pesq_5)
# stoi_sep_05 = read_snr_data(stoi_5)
#
# c = shelve.open(os.path.join(CRD_320_small,'score_0051.slv'))
# dns_6 = np.array(c['dns'])
# stoi_6 = np.array(c['STOI'])
# pesq_6 = np.array(c['pesq_torch'])
# si_sdr_6 = np.array(c['si_sdr'])
# b.close()
# dns_sep_06 = read_snr_data(dns_6)
# pesq_sep_06 = read_snr_data(pesq_6)
# stoi_sep_06 = read_snr_data(stoi_6)
# si_sdr_sep_6 = read_snr_data(si_sdr_6)
# print(pesq_sep_06)
# print(stoi_sep_06)
# print(si_sdr_sep_6)
# print(dns_sep_06)
c = shelve.open(os.path.join(CRD_320_opt,'score_0025.slv'))
#print(list(c.keys()))
dns_7 = np.array(c['dns'])
stoi_7 = np.array(c['STOI'])
pesq_7 = np.array(c['pesq_torch'])
si_sdr_7 = np.array(c['si_sdr'])
c.close()
dns_sep_07 = read_snr_data(dns_7)
pesq_sep_07 = read_snr_data(pesq_7)
stoi_sep_07 = read_snr_data(stoi_7)
si_sdr_sep_7 = read_snr_data(si_sdr_7)
print(pesq_sep_07)
print(stoi_sep_07)
print(si_sdr_sep_7)
print(dns_sep_07)

import pandas as pd
#
# data_result = {'SNR':[str('-5dB'),str('0dB'),str('5dB'),str('10dB'),str('15dB'),str('20dB')]}
#
# # #    ,'Model Name': ['Input Noisy','CRUSE5-320-2xLSTM1-concat-skip [4](baseline)','CRUSE5-320-2xLSTM2-concat-skip ','CRUSE5-320-1xLSTM2-add-skip',
# #              'CRUSE4-320-1xGRU4-conv-skip [11]','CRD-512-2xGRU1','CRD-320-1xGRU2+2GRU','CRD-320-2xGRU2+1GRU']
#
# df = pd.DataFrame(data_result)
# #data_frame = pd.DataFrame(df, columns=['SNR', 'Model', 'PESQ','STOI','si-SDR','DNS-MOS'])
#
# print("********")
# print("The LaTeX table is:")
# print(df.to_latex(index=True))
# index = dns_1 - dns
# index2 = pesq_1 - pesq
# #index3 = stoi_1 - stoi
# index11 = dns_2 - dns_1
# index12 = pesq_2 - pesq_1
# #index13 = stoi_2 - stoi_1
# index14 = dns_3 - dns_2
# index15 = pesq_3 - pesq
# #index16 = stoi_3 - stoi_2
# for a, b,e, c, d in zip(index,index14,index11,index12,index15):
#     if a<b<e and c<d:
#         print('True')
#         print(np.where(index == a))
#         print(name[np.where(index == a)])
# print(name[index])
# print(name[index2])
# print(name[index3])







# dns_enhanced_sep = read_snr_data(dns_enhanced)
# stoi_enhanced_sep = read_snr_data(stoi_enhanced)
# pesq_enhanced_sep = read_snr_data(pes_enhanced)

# plot_bar(y1=pesq_sep,y2=pesq_sep_1, y3=pesq_sep_model2_1,y4=pesq_sep_03, label='PESQ')
# plot_bar(y1=dns_sep,y2=dns_sep_1, y3=dns_sep_model2_1,y4=dns_sep_03, label='DNS')
# plot_bar(y1=stoi_sep,y2=stoi_sep_1, y3=stoi_sep_model2_1,y4=stoi_sep_03, label='STOI')






#
# for i in range(len(pathlist1)):
#     g = shelve.open(pathlist1[i][:-4])
#     klist = list(g.keys())
#     print(klist)
#     name = np.array(g['name'])
#     pes_enhanced = np.array(g['pesq_torch'])
#     dns_enhanced = np.array(g['dns'])
#     stoi_enhanced = np.array(g['STOI'])
#     LLRs_enhanced  =np.array(g['LLR'])
#     fwSNRsegs_enhanced = np.array(g['fwSNRseg'])
#     si_sdrs_enhanced = np.array(g['si_sdr'])
#     Q_enhanced = np.array(g['Quality'])
#     print('The Models name is:%s'% str(os.path.basename(pathlist1[i][:-8])))
#     print('     DNS: %s' % np.mean(dns_enhanced))
#     print('     PESQ: %s'% np.mean(pes_enhanced))
#     print('     STOI: %s'% np.mean(stoi_sep))
#     print('     LLR: %s' % np.mean(LLRs_enhanced))
#     print('     fwSNRseg: %s' % np.mean(fwSNRsegs_enhanced))
#     print('     si_sdrs: %s' % np.mean(si_sdrs_enhanced))
#     print('     Quality of Model: %s' % np.mean(Q_enhanced))
#     print('__________________******************************__________________')
#         dns_enhanced_sep = read_snr_data(dns_enhanced)
#         stoi_enhanced_sep = read_snr_data(stoi_enhanced)
#         pesq_enhanced_sep = read_snr_data(pes_enhanced)
#         index = np.argmax(dns_enhanced - dns)
#         print(name[index])
#         plot_bar(y1=pesq_sep, y3=pesq_enhanced_sep, label='PESQ')
#         plot_bar(y1=dns_sep, y3=dns_enhanced_sep, label='DNS')
#         plot_bar(y1=stoi_sep, y3=stoi_enhanced_sep, label='STOI')
#
#
#     delta_pes = np.array(g['delta_pesq_torch'])
#     delta_stoi = np.array(g['delta_STOI'])
#     delta_LLR = np.array(g['delta_LLR'])
#     delta_fwSNRseg = np.array(g['delta_fwSNRseg'])
#     delta_si_sdr = np.array(g['delta_si_sdr'])
#     delta_qaulity = np.array(g['delta_quality'])
#     delta_mos = np.array(g['delta_mos'])
#     print('     Delta MOS %s' % np.mean(delta_mos))
#     print('     Delta PESQ %s'% np.mean(delta_pes))
#     print('     Delta STOI %s'% np.mean(delta_stoi))
#     print('     Delta SNRseg %s' % np.mean(delta_LLR))
#     print('     Delta fwSNRseg %s' % np.mean(delta_fwSNRseg))
#     print('     Delta si_sdr %s' % np.mean(delta_si_sdr))
#     print('     Delta delta_qaulity %s' % np.mean(delta_qaulity))
#     print('__________________##############################__________________')



