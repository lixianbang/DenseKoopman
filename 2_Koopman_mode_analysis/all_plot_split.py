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

# 3D线图  x,y,time
def line_3d_xytime(split_num):
    # 线
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # c颜色，marker：样式*雪花
    for id in range(all_x.shape[0]):
        yuan_data_one = yuan_data[id]
        x_plot = yuan_data_one[:, 0]
        z_plot = yuan_data_one[:, 1]
        y_plot = np.arange(all_x.shape[1])  # time
        if x_plot[0] > x_plot[-1]:
            ax.plot(xs=x_plot, ys=y_plot, zs=z_plot, c="r", marker="*", markersize=2.2)
        else:
            ax.plot(xs=x_plot, ys=y_plot, zs=z_plot, c="g", marker="+", markersize=2.2)
    plt.savefig(f'./pic/all_traj_split_t/{split_num}.jpg')
    # plt.show()

# 3D线图  x/y  ,  time  ,  ID
def line_3d_val_time_id():
    # 线
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # c颜色，marker：样式*雪花
    for id in range(all_x.shape[0]):
        yuan_data_one = yuan_data[id]
        num_list = [id] * all_x.shape[1]
        x_plot = yuan_data_one[:, 0]
        z_plot = yuan_data_one[:, 1]
        y_plot = np.arange(all_x.shape[1])  # time
        if x_plot[0] > x_plot[-1]:
            ax.plot(xs=num_list, ys=y_plot, zs=z_plot, c="r", marker="*", markersize=2)
        else:
            ax.plot(xs=num_list, ys=y_plot, zs=z_plot, c="g", marker="+", markersize=2)

    plt.show()


# 分层轨迹数据导入
for num in range(78):
    # 分层层数
    split_num = num
    # 原轨迹数据导入 x
    all_x = np.loadtxt(f'plot_spaces/208_sp_{split_num}_x_HT21-03.txt')
    # 原轨迹数据导入 y
    all_y = np.loadtxt(f'plot_spaces/208_sp_{split_num}_y_HT21-03.txt')

    # print(all_x.shape, all_y.shape)
    yuan_data = np.zeros((all_x.shape[0], all_x.shape[1], 2))  # ID Frame (X,Y)

    for i in range(all_x.shape[0]):
        for j in range(all_x.shape[1]):
            yuan_data[i][j][0] = all_x[i][j]
            yuan_data[i][j][1] = all_y[i][j]
    # print(yuan_data)
    # line_3d_xytime(split_num)
    # line_3d_val_time_id()
