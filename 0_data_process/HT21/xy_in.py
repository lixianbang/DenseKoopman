import numpy as np
import os

data_path_x = './window_x/'
for file in os.listdir(data_path_x):
    file_path_x = os.path.join(data_path_x, file)
    file_path_y = file_path_x.replace('_x', '_y')
    file_save = file.replace('window_x_', '').replace('.txt', '')
    # print(file_save)

    Position_x = np.loadtxt(file_path_x, dtype=float)
    Position_y = np.loadtxt(file_path_y, dtype=float)

    hang = Position_x.shape[0]   # 20
    lie = int(Position_x.shape[1]/10)   # 20
    new_Position_x = np.zeros((hang, lie))
    new_Position_y = np.zeros((hang, lie))
    for jnew in range(Position_x.shape[1]):
        for inew in range(Position_x.shape[0]):
            if (jnew + 1) % 10 == 0:
                new_Position_x[inew][int((jnew + 1) / 10) - 1] = Position_x[inew][jnew]
                new_Position_y[inew][int((jnew + 1) / 10) - 1] = Position_y[inew][jnew]

    # print(num)
    print(new_Position_x)
    print(new_Position_y)

    middata = np.zeros((new_Position_x.shape[0] * new_Position_x.shape[1], 4))

    for i in range(middata.shape[0]):
        middata[i][0] = i // new_Position_x.shape[0]
        middata[i][1] = i % new_Position_x.shape[0] + 1
        middata[i][2] = new_Position_x[i % new_Position_x.shape[0]][i // new_Position_x.shape[0]]
        middata[i][3] = new_Position_y[i % new_Position_x.shape[0]][i // new_Position_x.shape[0]]

    out_path = f'./MID_data/mid_data_{file_save}_{new_Position_x.shape[0]}x{new_Position_x.shape[1]}.txt'
    np.savetxt(out_path, middata, fmt='%d')
print('All Done !')

