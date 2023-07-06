# import matplotlib.pyplot as plt
# import pandas as pd
#
# df = pd.DataFrame([['-5dB', 10, 20, 10, 30], [ '0dB', 20, 25, 15, 25], ['5dB', 12, 15, 19, 6],
#                    ['10dB', 10, 29, 13, 19],
#                    ['15dB', 10, 29, 13, 19],
#                    ['20dB', 10, 29, 13, 19]],
#                   columns=['Team', 'Round 1', 'Round 2', 'Round 3', 'Round 4'])
# print(df)
#
# df.plot(x='Team',
#         kind='bar',
#         stacked=False,
#         title='Grouped Bar Graph with dataframe')

import matplotlib.pyplot as plt
import numpy as np
import random

x1 = np.arange(6)
y1 = [34, 56, 12, 89, 67,1]
y2 = [12, 56, 78, 45, 90,1]
y3 = [14, 23, 45, 25, 89,1]
width = 0.1


def plot_bar(xx=x1, y1=y1,y2=y2, y3=y3,y4=y3,label='DNS'):

    #color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    plt.bar(xx - 0.15, y1, width, color=['black'])
    #color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    plt.bar(xx - 0.05, y2, width, color=['green'])
    #color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    plt.bar(xx + 0.05, y3, width, color=['red'])
    #color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    plt.bar(xx + 0.15, y4, width, color=['cyan'])
    plt.xticks(xx, ['-5dB', '0dB', '5dB', '10dB', '15dB','20dB'])
    plt.xlabel(label)
    plt.ylabel("Scores")
    plt.legend(["Noisy Signal", "CRUSE320-2xLSTM1-concate skip","cruse320-2xLSTM2-concate skip","cruse320-2xLSTM2-add Skip"])
    plt.show()


def read_snr_data(x):
    x_minus_5 = np.mean(x[0::6])
    x_0 = np.mean(x[1::6])
    x_10 = np.mean(x[2::6])
    x_15 = np.mean(x[3::6])
    x_20 = np.mean(x[4::6])
    x_5 = np.mean(x[5::6])
    return [x_minus_5, x_0, x_5, x_10, x_15, x_20]
