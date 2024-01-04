import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(8,6))
ax3d = plt.axes(projection="3d")

xdata = np.linspace(-3, 3, 100)
ydata = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(xdata, ydata)
Z = 1/(1+np.exp(-X-Y))

ax3d = plt.axes(projection='3d')
ax3d.plot_surface(X, Y, Z, cmap='plasma')
ax3d.set_title('Surface Plot in Matplotlib')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')

plt.show()