import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(-1, 1, N)
Y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([-0.1, 0.5])
sigma_x = 0.3
sigma_y = 0.2
corr = 0
Sigma = np.array([[sigma_x**2 , corr], [corr,  sigma_y**2]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# The distribution on the variables X, Y packed into pos.
F = multivariate_normal(mu, Sigma)
ean = [0, 0]
#Z = multivariate_gaussian(pos, mu, Sigma)
Z = F.pdf(pos)
# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
#                cmap=cm.viridis)
ax = fig.add_subplot(111)
cax  = ax.contourf(X, Y, Z, 50)
cbar = fig.colorbar(cax)
# Adjust the limits, ticks and view angle
#ax.set_zlim(-1,1)
#ax.set_zticks(np.linspace(0,0.2,5))
#ax.view_init(90, 0)

plt.show()
