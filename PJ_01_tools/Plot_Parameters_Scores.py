import matplotlib.pyplot as plt


# this file is under construction and not for the general use!
def plot_score_arrays(x_data, y_data, labels):
    if len(x_data) != len(y_data):
        print('different sizes')
        return

    # color and form definition
    #forms = ["v", "2", "*", 'P', "x", '^']
    forms = ['o', 'v', '^', 's', 'P', '*', '+', 'x']
    colors = ['b', 'r', 'y','k','c','m','k']
    markers = []
    # for j in range(len(colors)):
    #     for k in range(len(forms)):
    #         markers.append(colors[j] + forms[j])
    # # print(markers)
    from random import choice

    #combination = (choice(colors)+choice(forms))
    plt.figure()
    ax = plt.subplot(111)
    for i in range(len(x_data)):
        combination = (choice(colors) + choice(forms))
        plt.plot(y_data[i], x_data[i], combination, label=labels[i],markersize=25)
    #
    plt.legend(loc='lower left', bbox_to_anchor=(0.7, 0.05),
              ncol=1, fancybox=True, shadow=True, prop={'size': 40, 'family':'Times New Roman'})

    # Shrink current x-axis by 20%
    box = ax.get_position()
    ax.tick_params(axis='both', labelsize=40)
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    ax.set_xlabel('computational complexity[M]', fontsize=45, fontfamily="Times New Roman")
    ax.set_ylabel('quality improvement (Δ ζ)',  fontsize=45, fontfamily="Times New Roman")
    for label in ax.get_xticklabels():
        label.set_font_properties({'family':"Times New Roman", 'size': 45})
    for label in ax.get_yticklabels():
        label.set_font_properties({'family': "Times New Roman", 'size': 45})
    #ax.set_xticklabels(ax.get_xticks(),fontfamily="Times New Roman")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axis('auto')
    ax.set_axisbelow(False)

    plt.show()
    # plt.savefig(os.path.join(os.path.dirname(current_results_dir), 'Loss_Pesq_training_results_' + str(i) + '.png'))
    plt.close()


if __name__ == '__main__':
    # x_data = [1.746, 1.709, 1.285, 1.562, 3.006,3.025, 3.422, 5.922, 2.516, 4.551, 3.5, 4.881, 2.922, 1, 3.551]
    # y_data = [17.58, 17.31, 21.39, 17.35, 10, 13.15, 21, 32, 14, 13, 14,
    #           14, 21, 21, 11]
    # labels = ['Cruse5-Softplus-concat-skip-connection', 'Cruse5-Softplus-add-skip-connection',
    #           'Cruse4-Softplus-concat-skip-connection', 'Cruse5-Relu-add-skip-connction', 'Cruse5-Prelu_0,25-add-skip-connction', 'Cruse5-Prelu_0,25_2Bdir(GRU)_instead of LSTM',
    #           'Cruse5-Prelu_0,25_2Bdir(LSTM)_instead of LSTM', 'net_8', 'net_9',
    #           'net_10', 'net_11', 'net_12', 'net_13', 'net_14', 'net_15']
    x_data = [0.005,1.16,2.375,2.759,3.778,3.557, 4.16,4.48,4.21, 4.33]
    y_data =  [0,17.58,17.352,30.03, 30.14, 9.94, 29.44,117.59 ,17.17, 3.73]
    labels =['Wiener Filter','CRN5-320-baseline','CRN5-320-PReLU-add-skips','CRBDIRN5-320-add-skips',
             'CRBDIRN5-320-descending-skips','CRBDIRN6-320-descending-skips','CRBDIRN4-512-C={16,16,32,64}-descending-skips',
             'CRBDIRN4-512-C={16,16,32,128}-descending-skips', 'CRBDIRN5-512-C={16,32,64,128,128}-descending-skips', 'CRBDIRN6-512-C={16,32,64,128,256,128}-descending-skips'
    #     x_data = [0.005,1.16,2.375,2.759,3.778,3.557, 4.16,4.28,4.21, 4.21, 4.33]
             #     y_data =  [0,17.58,17.352,30.03, 30.14, 9.94, 29.44,117.59 ,17.17, 19.53, 3.73]
             # labels =['Wiener Filter','CRN5-320-Softplus-concat skips','CRN5-320-PReLU-add-skips','CRBDIRN5-320-add-skips',
    #          'CRBDIRN5-320-descending-skips','CRBDIRN6-320-descending-skips','CRBDIRN4-512-C={16,16,32,64}-descending-skips',
    #          'CRBDIRN4-512-C={16,16,32,128}-descending-skips', 'CRBDIRN5-512-C={16,32,64,128,128}-descending-skips',
    #          'CRBDIRN6-512-C={16,32,64,128,256,256}-descending-skips', 'CRBDIRN6-512-C={16,32,64,128,256,128}-descending-skips'
]
    plot_score_arrays(x_data, y_data, labels)
