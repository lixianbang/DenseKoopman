import matplotlib.pyplot as plt
import numpy as np
import math
def trans_num(str_num):
    before_e = float(str_num.split('e')[0])
    sign = str_num.split('e')[1][:1]
    after_e = int(str_num.split('e')[1][1:])

    if sign == '+':
        float_num = before_e * math.pow(10, after_e)
    elif sign == '-':
        float_num = before_e * math.pow(10, -after_e)
    else:
        float_num = None
        print('error: unknown sign')
    return float_num


# 分层轨迹数据导入
yuan_data = np.zeros((20, 20, 2))  # ID Frame (X,Y)
split_num = 6
with open(f'fenceng/tab_mid_{split_num}_MOT20-01_19_20x20.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip('\n')
        line_list = line.split('\t')
        # print(line_list)
        ID_ = int(trans_num(line_list[1])) - 1
        Frame_ = int(float(line_list[0]) * 0.1)
        # print(ID_, Frame_)
        yuan_data[ID_][Frame_][0] = trans_num(line_list[2])
        yuan_data[ID_][Frame_][1] = trans_num(line_list[3])

# 3D线图
def line_3d():
    # 线
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # c颜色，marker：样式*雪花
    for id in range(20):
        if id == 0:
            yuan_data_one = yuan_data[id]
            x_plot = yuan_data_one[:, 0]
            z_plot = yuan_data_one[:, 1]
            y_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # time
            ax.plot(xs=x_plot, ys=z_plot, zs=y_plot, c="g", marker="+", markersize=5)
        if id == 1:
            yuan_data_one = yuan_data[id]
            x_plot = yuan_data_one[:, 0]
            z_plot = yuan_data_one[:, 1]
            y_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # time
            ax.plot(xs=x_plot, ys=z_plot, zs=y_plot, c="r", marker="+", markersize=5)
    plt.show()
# line_3d()


# 3D线图  x/y  ,  time  ,  ID
def line_3d_val_time_id():
    # 线
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # c颜色，marker：样式*雪花
    for id in range(20):
        yuan_data_one = yuan_data[id]
        num_list = [id, id, id, id, id, id, id, id, id, id, id, id, id, id, id, id, id, id, id, id]
        x_plot = yuan_data_one[:, 0]
        z_plot = yuan_data_one[:, 1]
        y_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # time
        ax.plot(xs=num_list, ys=y_plot, zs=z_plot, c="g", marker="+", markersize=5, linestyle='-')
    plt.show()
line_3d_val_time_id()