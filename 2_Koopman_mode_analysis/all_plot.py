import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 原轨迹数据导入 x
all_x = np.loadtxt('plot/208_window_x_HT21-03.txt')
# 原轨迹数据导入 y
all_y = np.loadtxt('plot/208_window_y_HT21-03.txt')

print(all_x.shape, all_y.shape)

yuan_data = np.zeros((all_x.shape[0], all_x.shape[1], 2))  # ID Frame (X,Y)

for i in range(all_x.shape[0]):
    for j in range(all_x.shape[1]):
        yuan_data[i][j][0] = all_x[i][j]
        yuan_data[i][j][1] = all_y[i][j]
print(yuan_data)

# 3D线图  x,y,time
def line_3d_xytime():
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
    plt.show()
line_3d_xytime()


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
            ax.plot(xs=num_list, ys=y_plot, zs=x_plot, c="r", marker="*", markersize=2)
        else:
            ax.plot(xs=num_list, ys=y_plot, zs=x_plot, c="g", marker="+", markersize=2)

    plt.show()
# line_3d_val_time_id()

# 3D曲面