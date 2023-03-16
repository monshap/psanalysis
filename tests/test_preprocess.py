import os
import sys

import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import skew

sys.path.append(os.path.abspath("../"))
import psanalysis.analyze_grids as ag

wd = os.path.dirname(os.path.realpath(__file__))
tpts = np.arange(0, 80, 2)
test_dir = ag.PlanarStudy(os.path.join(wd, "SampleImages", "Subject1"), tpts)
test_dir.preprocess_scans(debug=False)
(a, grid_aTc, grid_aIn, grid_pTc, grid_pIn,
 g2p_aTc, g2p_aIn, g2p_pTc, g2p_pIn) = test_dir.gridify(16, 8, keeppixels=True)

# Calculate pixel statistics
g2p_gmTc = gmean(np.clip(np.stack((g2p_aTc, g2p_pTc)), 1e-6, None))
g2p_gmIn = gmean(np.clip(np.stack((g2p_aIn, g2p_pIn)), 1e-6, None))
cROI = np.zeros((16, 8))
cROI[4:12, 0:4] = 1
pROI = 1 - cROI
cidx = np.nonzero(cROI)
pidx = np.nonzero(pROI)

# Calculate mean
# central grids
cgrid_mean = np.mean(np.nanmean(g2p_gmTc[cidx[0], cidx[1], ..., 0],
                     axis=(1, 2)))
# central ROI
cROI_mean = np.nanmean(g2p_gmTc[cidx[0], cidx[1], ..., 0])
# peripheral grids
pgrid_mean = np.mean(np.nanmean(g2p_gmTc[pidx[0], pidx[1], ..., 0],
                     axis=(1, 2)))
# peripheral ROI
pROI_mean = np.nanmean(g2p_gmTc[pidx[0], pidx[1], ..., 0])

# Calculate variance
# central grids
cgrid_var = np.mean(np.nanvar(g2p_gmTc[cidx[0], cidx[1], ..., 0],
                     axis=(1, 2)))
# central ROI
cROI_var = np.nanvar(g2p_gmTc[cidx[0], cidx[1], ..., 0])
# peripheral grids
pgrid_var = np.mean(np.nanvar(g2p_gmTc[pidx[0], pidx[1], ..., 0],
                     axis=(1, 2)))
# peripheral ROI
pROI_var = np.nanvar(g2p_gmTc[pidx[0], pidx[1], ..., 0])

# Calculate skew
# central grids
cgrid_skew = np.mean([skew(g2p_gmTc[cidx[0][i], cidx[1][i], ..., 0],
                           nan_policy="omit") for i in range(len(cidx[0]))])
# central ROI
cROI_skew = skew(g2p_gmTc[cidx[0], cidx[1], ..., 0], axis=None,
                 nan_policy="omit")
# peripheral grids
pgrid_skew = np.mean([skew(g2p_gmTc[pidx[0][i], pidx[1][i], ..., 0],
                           nan_policy="omit") for i in range(len(pidx[0]))])
# peripheral ROI
pROI_skew = skew(g2p_gmTc[pidx[0], pidx[1], ..., 0], axis=None,
                 nan_policy="omit")

prows = [s.center(6) for s in ["cgrid", "cROI", "pgrid", "pROI"]]
print("| Region | Mean | Variance | Skew  |")
print(36*"-")
print(f"| {prows[0]} | {cgrid_mean:>4.2f} | {cgrid_var:>8.2f} | {cgrid_skew:>5.2f} |")
print(f"| {prows[1]} | {cROI_mean:>4.2f} | {cROI_var:>8.2f} | {cROI_skew:>5.2f} |")
print(f"| {prows[2]} | {pgrid_mean:>4.2f} | {pgrid_var:>8.2f} | {pgrid_skew:>5.2f} |")
print(f"| {prows[3]} | {pROI_mean:>4.2f} | {pROI_var:>8.2f} | {pROI_skew:>5.2f} |")
