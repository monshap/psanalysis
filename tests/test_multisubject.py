import os
import sys

import numpy as np
from scipy.stats.mstats import gmean

sys.path.append(os.path.abspath("../"))
import psanalysis.analyze_grids as ag

wd = os.path.dirname(os.path.realpath(__file__))
tpts = np.arange(0, 80, 2)
dir_list = [f"Subject{i}" for i in range(1, 4)]
(ny, nx) = (16, 8)
nt = len(tpts)
nsub = len(dir_list)
# Initialize empty arrays
all_areas = np.zeros((nsub,))
all_aTc = np.zeros((ny, nx, nt, nsub))
all_pTc = np.zeros((ny, nx, nt, nsub))
all_aIn = np.zeros((ny, nx, nt, nsub))
all_pIn = np.zeros((ny, nx, nt, nsub))
# Fill with activity per grid for each subject
for i, sub in enumerate(dir_list):
    test_dir = ag.PlanarStudy(os.path.join(wd, "SampleImages", sub), tpts)
    test_dir.preprocess_scans(debug=False)
    (a, grid_aTc, grid_aIn, grid_pTc, grid_pIn) = test_dir.gridify(ny, nx)
    all_areas[i] = a
    all_aTc[..., i] = grid_aTc
    all_pTc[..., i] = grid_pTc
    all_aIn[..., i] = grid_aIn
    all_pIn[..., i] = grid_pIn
# Calculate the geometric mean of anterior & posterior images
Tc_stack = np.clip(np.stack((all_aTc, all_pTc)), 0, None)
In_stack = np.clip(np.stack((all_aIn, all_pIn)), 0, None)
gm_Tc = gmean(Tc_stack, axis=0)
gm_In = gmean(In_stack, axis=0)
