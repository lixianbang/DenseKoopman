import numpy as np

data_name = 'HT21-03'
middle_data_path_x = f'position_x_{data_name}.txt'
middle_data_path_y = f'position_y_{data_name}.txt'
middle_data_x = np.loadtxt(middle_data_path_x, dtype=float)
middle_data_y = np.loadtxt(middle_data_path_y, dtype=float)

# 帧数 ID Xmin Ymin width height flag category visibility
row = middle_data_x.shape[0]
column = middle_data_x.shape[1]  # x、y的行列数应一致
# print(middle_data_x.shape)
max = 20
for j in range(199, middle_data_x.shape[1]):
    data_split_x = middle_data_x[:, j - 199: j + 1]
    data_window_x = np.zeros((20, 200))

    data_split_y = middle_data_y[:, j - 199: j + 1]
    data_window_y = np.zeros((20, 200))

    # 对data_split进行遍历，查找0值并舍弃
    dele = []
    for ii in range(data_split_x.shape[0]):
        for jj in range(data_split_x.shape[1]):
            if data_split_x[ii][jj] == 0:
                dele.append(ii)
                break
    deniose_data_split_x = np.delete(data_split_x, dele, 0)
    deniose_data_split_y = np.delete(data_split_y, dele, 0)   # 去除含有0的行，此时应为全满数组
    # print(deniose_data_split_x.shape)

    if deniose_data_split_x.shape[0] < max:
        continue
    # 以[20, 200]窗口取数据
    else:
        max = deniose_data_split_x.shape[0]
        data_window_x = deniose_data_split_x
        data_window_y = deniose_data_split_y
        data_window_x_out_path = f'./plot/{j}_window_x_{data_name}.txt'
        data_window_y_out_path = f'./plot/{j}_window_y_{data_name}.txt'
        np.savetxt(data_window_x_out_path, data_window_x, fmt='%d')
        np.savetxt(data_window_y_out_path, data_window_y, fmt='%d')
        print(max)








