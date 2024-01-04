# coding=gbk
import numpy as np
from GenerateKoopmanModes import GenerateKoopmanModes as GKM
import os

data_path_x = './window_y/'
for file in os.listdir(data_path_x):
    file_path_x = os.path.join(data_path_x, file)

    raw_data = np.loadtxt(file_path_x, dtype=float)
    num = raw_data.shape[0]
    time = raw_data.shape[1]
    # print(num)
    # print(time/10)
    new_data = np.zeros((num, int(time/10)))
    for jnew in range(time):
        for inew in range(num):
            if (jnew+1) % 10 == 0:
                new_data[inew][int((jnew+1)/10)-1] = raw_data[inew][jnew]


    # print(num)
    # print(new_data)
    space_num = 19
    #
    # print(num, time)
    Psi = GKM(new_data, delay=1, delt=1, mode1=1, mode2=space_num)
    psi = np.zeros((space_num, num, int(time/10)))

    flag = 0
    for k in range(space_num):
        for i in range(num):
            for j in range(int(time/10)):
                psi[k][i][j] = float(Psi[i][j][k])
                if psi[k][i][j]**2 == 0:
                    flag += 1
    if flag == 0:
        for k in range(space_num):
            path_out = './19spaces/win_y/' + file.replace('window', f'mid_{k}')
            np.savetxt(path_out, psi[k])


