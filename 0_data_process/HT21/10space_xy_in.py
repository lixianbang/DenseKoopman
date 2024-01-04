import numpy as np
import os

data_path_x = './19spaces/win_x'
for file in os.listdir(data_path_x):
    try:
        file_path_x = os.path.join(data_path_x, file)
        file_path_y = file_path_x.replace('_x', '_y')
        file_save = file.replace('x_', '').replace('.txt', '')
        # print(file_save)

        Position_x = np.loadtxt(file_path_x, dtype=float)
        Position_y = np.loadtxt(file_path_y, dtype=float)

        hang = Position_x.shape[0]   # 20
        lie = Position_x.shape[1]   # 200
        middata = np.zeros((Position_x.shape[0] * Position_x.shape[1], 4))

        for i in range(middata.shape[0]):
            middata[i][0] = i // Position_x.shape[0]
            middata[i][1] = i % Position_x.shape[0] + 1
            middata[i][2] = Position_x[i % Position_x.shape[0]][i // Position_x.shape[0]]
            middata[i][3] = Position_y[i % Position_x.shape[0]][i // Position_x.shape[0]]

        out_path = f'./19spaces/MID_data/{file_save}_{Position_x.shape[0]}x{Position_x.shape[1]}.txt'
        np.savetxt(out_path, middata, fmt='%f', delimiter='\t')
    except:
        print('not done')
print('All Done !')

