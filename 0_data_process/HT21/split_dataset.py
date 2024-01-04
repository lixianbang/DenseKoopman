import numpy as np
import os

train_name = []
test_name = []
for i in range(4):
    for j in range(19, 162):
        if i == 0:  # HT01
            if j < 21: test_name.append(f'01_{j}_20')
            else: train_name.append(f'01_{j}_20')
        elif i == 1: # HT02
            if j < 46: test_name.append(f'02_{j}_20')
            else: train_name.append(f'02_{j}_20')
        elif i == 2:  # HT03
            if j < 50: test_name.append(f'03_{j}_20')
            else: train_name.append(f'03_{j}_20')
        elif i == 3:  # HT04
            if j < 39: test_name.append(f'04_{j}_20')
            else: train_name.append(f'04_{j}_20')
data_path = './19spaces/MID_data'
save_path_train = './19spaces/19train'
save_path_test = './19spaces/19test'
for file in os.listdir(data_path):
    file_0 = file.split('HT21-')
    file_1 = file_0[1].split('x20')
    file_name = file_1[0]
    # print(file_name)
    file_path = os.path.join(data_path, file)
    Position = np.loadtxt(file_path, dtype=float)
    for name_train in train_name:
        if file_name in name_train:
            np.savetxt(os.path.join(save_path_train, file), Position, fmt='%f', delimiter='\t')
    for name_test in test_name:
        if file_name in name_test:
            np.savetxt(os.path.join(save_path_test, file), Position, fmt='%f', delimiter='\t')



