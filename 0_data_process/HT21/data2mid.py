import numpy as np

# data_name = 'HT21-01'
# data_path = f'./using/{data_name}/gt/gt.txt'
# Data = np.loadtxt(data_path, dtype=float)
# # 帧数 ID Xmin Ymin width height flag category visibility
# hang = Data.shape[0]
# lie = Data.shape[1]
# print(Data.shape)
#
# # 寻找大矩阵的行数和列数，即为最大ID数量和最长延续时间
# min_ID = Data[0][1]
# max_ID = Data[0][1]
# min_T = Data[0][0]
# max_T = Data[0][0]
# for ix in range(hang):
#     min_ID = min(min_ID, Data[ix][1])
#     max_ID = max(max_ID, Data[ix][1])
#
#     min_T = min(min_T, Data[ix][0])
#     max_T = max(max_T, Data[ix][0])
# print(min_ID, max_ID, min_T, max_T)
#
# T = int(max_T - min_T + 2)  # 大矩阵的列数
# ID_num = int(max_ID - min_ID + 2)  # 大矩阵的行数
#
# Position_X = np.zeros((ID_num, T))
# Position_Y = np.zeros((ID_num, T))
#
# for i in range(hang):
#     if Data[i][7] == 1:  # 此时为行人的类别
#         frame_num = Data[i][0]
#         ID = Data[i][1]
#         # 求当前行人的x、y
#         positon_x = Data[i][2] + (Data[i][4]) / 2
#         positon_y = Data[i][3] + (Data[i][5]) / 2
#         # 此时输出坐标
#         Position_X[int(ID-min_ID)][int(frame_num-min_T)] = positon_x
#         Position_Y[int(ID-min_ID)][int(frame_num-min_T)] = positon_y
#
# x_out_path = f'position_x_{data_name}.txt'  # middle_data_path_x
# y_out_path = f'position_y_{data_name}.txt'  # middle_data_path_y
# np.savetxt(x_out_path, Position_X, fmt='%d')
# np.savetxt(y_out_path, Position_Y, fmt='%d')
# print('All Done !')



###=====================================================================================================================

data_name = 'HT21-04'
middle_data_path_x = f'position_x_{data_name}.txt'
middle_data_path_y = f'position_y_{data_name}.txt'
middle_data_x = np.loadtxt(middle_data_path_x, dtype=float)
middle_data_y = np.loadtxt(middle_data_path_y, dtype=float)

# 帧数 ID Xmin Ymin width height flag category visibility
row = middle_data_x.shape[0]
column = middle_data_x.shape[1]  # x、y的行列数应一致
print(middle_data_x.shape)

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
    print(deniose_data_split_x.shape)
    if deniose_data_split_x.shape[0] < 20:
        continue
    # 以[20, 200]窗口取数据
    else:
        for iv in range(19, deniose_data_split_x.shape[0]):
            data_window_x = deniose_data_split_x[iv - 19: iv + 1, :]
            data_window_y = deniose_data_split_y[iv - 19: iv + 1, :]
            data_window_x_out_path = f'./window_x/window_x_{data_name}_{iv}.txt'  # middle_data_path_x
            data_window_y_out_path = f'./window_y/window_y_{data_name}_{iv}.txt'  # middle_data_path_y
            np.savetxt(data_window_x_out_path, data_window_x, fmt='%d')
            np.savetxt(data_window_y_out_path, data_window_y, fmt='%d')








