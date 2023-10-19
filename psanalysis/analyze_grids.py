import os

import numpy as np


def bin_stats(grid):
    # calculate first three moments of pixels in grid
    from scipy.stats import skew

    bin_mean = np.mean(grid, axis=(0, 1))
    bin_var = np.var(grid, axis=(0, 1))
    bin_skew = skew(grid, axis=(0, 1))
    return bin_mean, bin_var, bin_skew


def pixel2grid(img, ny, nx, keeppixels=False):
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
    xfs = np.floor(x_lin)
    x_flr = [int(xf) for xf in xfs]
    yfs = np.floor(y_lin)
    y_flr = [int(yf) for yf in yfs]
    area = x_lin[1] * y_lin[1]
    y_max = np.max(yfs[1:] - yfs[:-1] + 1)
    x_max = np.max(xfs[1:] - xfs[:-1] + 1)
    # dimensions of individual bin
    Bdims = np.copy(fdims)
    By = int(y_max)
    Bx = int(x_max)
    Bdims[0:2] = [By, Bx]
    # inverse lookup
    grid2pixels = np.full((ny, nx, *Bdims), np.nan)
    for i in range(ny):
        for j in range(nx):
            [x0, x1] = x_flr[j:j+2]
            [y0, y1] = y_flr[i:i+2]
            [lx0, lx1] = x_lin[j:j+2]
            [ly0, ly1] = y_lin[i:i+2]
            # create mask defining contribution from each pixel (activity
            # from pixels on bin edge is split proportionally to area)
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
            # activity from all whole & partial pixels in grid
            B = img[y0:y1+1, x0:x1+1, ...] * A
            if keeppixels:
                grid2pixels[i, j, :(y1 - y0 + 1), :(x1 - x0 + 1), ...] = B
            # add together all activity in grid
            binned[i, j, ...] = np.sum(B, axis=(0, 1))
    return binned, area, grid2pixels


class PlanarStudy(object):

    def __init__(self, dir, t_pts, name=None):
        self.dir = dir
        self.t_pts = t_pts
        self.wlmask = np.loadtxt(os.path.join(self.dir, "WLMask.txt"),
                                 dtype="bool")
        self.ant_dir = os.path.join(self.dir, "AntScans")
        self.post_dir = os.path.join(self.dir, "TcScans")
        self.ant_bg_dir = os.path.join(self.dir, "AntBgScans")
        self.post_bg_dir = os.path.join(self.dir, "BgScans")

    def preprocess_scans(self, debug=False, tc_only=False):

        def _clip_scan(scan, py0, px0, debug=False):
            py, px = np.shape(scan)
            if py > py0:
                dy = py - py0
                scan = scan[dy:, :]
                py = int(py0)
            if px > px0:
                dx = px - px0
                scan = scan[:, dx:]
                px = int(px0)
            if debug:
                print(f"Current (py, px): ({py}, {px})")
            return scan, py, px

        def bgcorrect_tc(act, bg, nt):
            tc_raw = act[..., :nt]
            mid_raw = act[..., nt:2*nt]
            tc_correct = tc_raw/2 - mid_raw*2.01/2 - bg[..., np.newaxis]/4
            return tc_correct

        def bgcorrect_in(act, bg, nt):
            in_raw = act[..., 2*nt:]
            in_correct = in_raw/2 - bg[..., np.newaxis]/4
            return in_correct

        nt = len(self.t_pts)
        py0, px0 = np.shape(self.wlmask)
        if debug:
            print(f"WLMask Shape: ({py0}, {px0})")

        # Process anterior scans
        ant_act = np.zeros((py0, px0, 3*nt))
        # Anterior indices for each energy window [Tc, Mid, In]
        ant_idx = [*range(1, nt+1), *range(2*nt+1, 3*nt+1),
                   *range(4*nt+1, 5*nt+1)]
        if tc_only:
            ant_idx = ant_idx[:2*nt]
        for i, idx in enumerate(ant_idx):
            fname = os.path.join(self.ant_dir, f"Tc_{idx}.txt")
            scan, py, px = _clip_scan(np.loadtxt(fname), py0, px0,
                                      debug=debug)
            ant_act[:py, :px, i] = scan
            ant_act[..., i] = ant_act[..., i]*self.wlmask
        # Anterior background Tc-99m activity
        ant_bgTc = np.zeros((py0, px0))
        ant_bgTc_path = os.path.join(self.ant_bg_dir, "Bg_1.txt")
        scan, py, px = _clip_scan(np.loadtxt(ant_bgTc_path), py0, px0,
                                  debug=debug)
        ant_bgTc[:py, :px] = scan
        ant_bgTc = ant_bgTc*self.wlmask
        # Anterior background In-111 activity
        if not tc_only:
            ant_bgIn = np.zeros((py0, px0))
            ant_bgIn_path = os.path.join(self.ant_bg_dir, "Bg_5.txt")
            scan, py, px = _clip_scan(np.loadtxt(ant_bgIn_path), py0,
                                      px0, debug=debug)
            ant_bgIn[:py, :px] = scan
            ant_bgIn = ant_bgIn*self.wlmask

        # Process posterior scans
        post_act = np.zeros((py0, px0, 3*nt))
        # Posterior indices for each energy window [Tc, Mid, In]
        post_idx = [*range(nt+1, 2*nt+1), *range(3*nt+1, 4*nt+1),
                    *range(5*nt+1, 6*nt+1)]
        if tc_only:
            post_idx = post_idx[:2*nt]
        for i, idx in enumerate(post_idx):
            fname = os.path.join(self.post_dir, f"Tc_{idx}.txt")
            scan, py, px = _clip_scan(np.loadtxt(fname), py0, px0,
                                      debug=debug)
            post_act[:py, :px, i] = scan
            post_act[..., i] = post_act[..., i]*self.wlmask
        # Posterior background Tc-99m activity
        post_bgTc = np.zeros((py0, px0))
        post_bgTc_path = os.path.join(self.post_bg_dir, "Bg_2.txt")
        scan, py, px = _clip_scan(np.loadtxt(post_bgTc_path), py0, px0,
                                  debug=debug)
        post_bgTc[:py, :px] = scan
        post_bgTc = post_bgTc*self.wlmask
        # Posterior background In-111 activity
        if not tc_only:
            post_bgIn = np.zeros((py0, px0))
            post_bgIn_path = os.path.join(self.post_bg_dir, "Bg_6.txt")
            scan, py, px = _clip_scan(np.loadtxt(post_bgIn_path), py0,
                                      px0, debug=debug)
            post_bgIn[:py, :px] = scan
            post_bgIn = post_bgIn*self.wlmask

        # Background correct anterior & posterior images
        ant_Tc = bgcorrect_tc(ant_act, ant_bgTc, nt)
        post_Tc = bgcorrect_tc(post_act, post_bgTc, nt)
        if tc_only:
            ant_In = np.zeros_like(ant_Tc)
            post_In = np.zeros_like(post_Tc)
        else:
            ant_In = bgcorrect_in(ant_act, ant_bgIn, nt)
            post_In = bgcorrect_in(post_act, post_bgIn, nt)

        # Assign attributes to activity matrices
        self.ant_Tc = ant_Tc
        self.ant_In = ant_In
        self.post_Tc = post_Tc
        self.post_In = post_In

        return ant_Tc, ant_In, post_Tc, post_In

    def gridify(self, ny, nx, keeppixels=False):
        emsg = "ERROR: No images loaded. Run preprocess_scans() first"
        assert hasattr(self, "ant_Tc"), emsg
        assert hasattr(self, "ant_In"), emsg
        assert hasattr(self, "post_Tc"), emsg
        assert hasattr(self, "post_In"), emsg

        grid_ant_Tc, area, g2p_ant_Tc = pixel2grid(self.ant_Tc, ny, nx,
                                                   keeppixels=keeppixels)
        grid_ant_In, _, g2p_ant_In = pixel2grid(self.ant_In, ny, nx,
                                                keeppixels=keeppixels)
        grid_post_Tc, _, g2p_post_Tc = pixel2grid(self.post_Tc, ny, nx,
                                                  keeppixels=keeppixels)
        grid_post_In, _, g2p_post_In = pixel2grid(self.post_In, ny, nx,
                                                  keeppixels=keeppixels)

        if keeppixels:
            return (area, grid_ant_Tc, grid_ant_In, grid_post_Tc, grid_post_In,
                    g2p_ant_Tc, g2p_ant_In, g2p_post_Tc, g2p_post_In)
        else:
            return (area, grid_ant_Tc, grid_ant_In, grid_post_Tc, grid_post_In)
