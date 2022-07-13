import numpy as np


def pixel2grid(img, ny, nx):
    # size of original image (could be more than 2 dim)
    sz = np.shape(img)
    h, w = sz[0:2]
    # size of final dimensions
    fdims = np.copy(sz)
    fdims[0:2] = [ny, nx]
    ndextra = img.ndim - 2
    # divide image into 2D bins
    binned = np.zeros(fdims)
    x_lin = np.linspace(0, w, nx+1)
    y_lin = np.linspace(0, h, ny+1)
    x_flr = np.floor(x_lin)
    y_flr = np.floor(y_lin)
    area = x_lin[1] * y_lin[1]
    for i in range(ny):
        for j in range(nx):
            [x0, x1] = x_flr[j:j+2]
            [y0, y1] = y_flr[i:i+2]
            [lx0, lx1] = x_lin[j:j+2]
            [ly0, ly1] = y_lin[i:i+2]
            Ay = y1 - y0 + 1
            Ax = x1 - x0 + 1
            Adims = np.copy(fdims)
            Adims[0:2] = [Ay, Ax]
            A = np.ones(Adims)
            A[:, 0, ...] = A[:, 0, ...] * (x0 - lx0 + 1)
            A[:, -1, ...] = A[:, -1, ...] * (lx1 - x1)
            A[0, ...] = A[0, ...] * (y0 - ly0 + 1)
            A[-1, ...] = A[-1, ...] * (ly1 - y1)
            if i == ny-1:
                A = A[0:-1, ...]
                y1 -= 1
            if j == nx-1:
                A = A[:, 0:-1, ...]
                x1 -= 1
            B = img[y0:y1+1, x0:x1+1, ...] * A
            binned[i, j, ...] = np.sum(B, axis=(0, 1))
    return binned, area
