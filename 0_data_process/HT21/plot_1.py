# coding=gbk
import numpy as np
from GenerateKoopmanModes import GenerateKoopmanModes as GKM
import os

data_path_x = './plot/'
for file in os.listdir(data_path_x):
    file_path_x = os.path.join(data_path_x, file)
    raw_data = np.loadtxt(file_path_x, dtype=float)

    num = raw_data.shape[0]
    time = raw_data.shape[1]
    space_num = raw_data.shape[0] - 1

    Psi = GKM(raw_data, delay=1, delt=1, mode1=1, mode2=space_num)
    psi = np.zeros((space_num, num, time))
    flag = 0

    for k in range(space_num):
        for i in range(num):
            for j in range(time):
                psi[k][i][j] = float(Psi[i][j][k])
                if psi[k][i][j]**2 == 0:
                    flag = 1
    if flag >= 0:
        print(file)
        for k in range(space_num):
            path_out = './plot_spaces/' + file.replace('window', f'sp_{k}')
            np.savetxt(path_out, psi[k])


