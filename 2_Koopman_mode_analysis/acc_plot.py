import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 两点欧式距离计算函数
def dis_com(x1, y1, x2, y2):
    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance
# print(vel_com(1, 2, 3, 4))

# 原轨迹数据导入 x
all_x = np.loadtxt('plot/208_window_x_HT21-03.txt')
# 原轨迹数据导入 y
all_y = np.loadtxt('plot/208_window_y_HT21-03.txt')
# print(all_x.shape, all_y.shape)

# 初始化轨迹记录矩阵
yuan_data = np.zeros((all_x.shape[0], all_x.shape[1], 2))  # ID Frame (X,Y)

# 轨迹矩阵赋值
for i in range(all_x.shape[0]):
    for j in range(all_x.shape[1]):
        yuan_data[i][j][0] = all_x[i][j]
        yuan_data[i][j][1] = all_y[i][j]
# print(yuan_data)

# 初始化加速度记录矩阵
data_acc = np.zeros((all_x.shape[0], all_x.shape[1]))  # ID Frame vel_value

# 速度矩阵赋值
for iv in range(all_x.shape[0]):
    for jv in range(all_x.shape[1]):
        if jv == 0:
            point_left_x = all_x[iv][jv]
            point_left_y = all_y[iv][jv]
            point_right_x = all_x[iv][jv + 1]
            point_right_y = all_y[iv][jv + 1]
            dis_ = dis_com(point_left_x, point_left_y, point_right_x, point_right_y)
            acc_ = dis_ / ((8 / all_x.shape[1]) ** 2)
            data_acc[iv][jv] = acc_
        elif jv == (all_x.shape[1]-1):
            point_left_x = all_x[iv][jv - 1]
            point_left_y = all_y[iv][jv - 1]
            point_right_x = all_x[iv][jv]
            point_right_y = all_y[iv][jv]
            dis_ = dis_com(point_left_x, point_left_y, point_right_x, point_right_y)
            acc_ = dis_ / ((8 / all_x.shape[1]) ** 2)
            data_acc[iv][jv] = acc_
        else:
            point_left_x = all_x[iv][jv - 1]
            point_left_y = all_y[iv][jv - 1]
            point_right_x = all_x[iv][jv + 1]
            point_right_y = all_y[iv][jv + 1]
            dis_ = dis_com(point_left_x, point_left_y, point_right_x, point_right_y)
            acc_ = dis_ / ((2 * (8 / all_x.shape[1])) ** 2)
            data_acc[iv][jv] = acc_
# print(data_vel)

# 绘制速度分布时空图
# 3D线图  x/y  ,  time  ,  ID
def line_3d_val_time_id():
    # 线
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # c颜色，marker：样式*雪花
    for id in range(all_x.shape[0]):
        x_plot = [id] * all_x.shape[1]
        z_plot = data_acc[id]
        y_plot = np.arange(all_x.shape[1])  # time
        # print(x_plot, y_plot, z_plot)
        if np.mean(z_plot) < 1000:
            ax.plot(xs=x_plot, ys=y_plot, zs=z_plot, c="g", marker="+", markersize=2)
        elif np.mean(z_plot) > 2000:
            ax.plot(xs=x_plot, ys=y_plot, zs=z_plot, c="r", marker="*", markersize=2)
        else:
            ax.plot(xs=x_plot, ys=y_plot, zs=z_plot, c="b", marker="o", markersize=2)
    plt.show()
line_3d_val_time_id()











