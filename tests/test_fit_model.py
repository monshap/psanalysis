import os
import sys
from math import log

import numpy as np
from scipy.io import loadmat

sys.path.append(os.path.abspath("../"))
import psanalysis.grid_model as gm


# define subject labels
sub_list = [f"Subject{i}" for i in range(1, 4)]
nsub = len(sub_list)

# gm_Tc format: (row, col, t, subject)
gm_Tc = np.load("MultiSub_Conc.npz")["gm_Tc"]
Tc0 = np.sum(gm_Tc[:,:,0,:], axis=(0,1))
Tc_norm = np.divide(gm_Tc, Tc0, out=np.zeros_like(gm_Tc), where=Tc0!=0)*100

nc0 = np.percentile(Tc_norm, 25, axis=2)  # non-clearable activity
clr = Tc_norm - nc0[:, :, np.newaxis, :]   # clearable activity
ny, nx, nt, nsub2 = np.shape(Tc_norm)
assert nsub2 == nsub, "Number of subjects don't match!"
ngrid = nx*ny
ts = list(range(0, 80, 2))

# define elevation map
el = np.load("Emap_16x8_diags_HRCT.npz")["arr_0"]
eq_thresh = 4
eq = el < eq_thresh
out_lung = el == 0
(bool_n, bool_ne, bool_e, bool_se,
 bool_s, bool_sw, bool_w, bool_nw) = gm.get_bool_maps(el, eq=eq)
sum_leave = np.sum([bool_n, bool_ne, bool_e, bool_se, bool_s, bool_sw,
                    bool_w, bool_nw], axis=0)

# for reduced model:
# n_clust = 5
# clmap_name = f"k{n_clust}_model"

# for full-scale model:
clmap_name = "full_model"

clust = np.load(f"{clmap_name}_clust.npz")["arr_0"]
clust = clust.astype("int32")
nout = np.sum(out_lung)
nin = ngrid - nout

# save outputs to file (False or name of output file)
save_params = f"{clmap_name}_params.npz"
save_grid_sse = f"{clmap_name}_grid_err.npz"

# load from previous run (False or name of input file)
load_params = False
load_grid_sse = False

all_ks = np.zeros((nK, nsub))
all_grid_sse = np.zeros((ny, nx, nsub))

if isinstance(load_params, str):
    all_ks = np.load(load_params)["all_ks"]
if isinstance(load_grid_sse, str):
    all_grid_sse = np.load(load_grid_sse)["all_grid_sse"]

for pat_num, pat_str in enumerate(sub_list):
    eq = el < eq_thresh
    pat_clr = clr[..., pat_num]
    pat_conc = Tc_norm[..., pat_num]
    print(pat_str)
    pat_gm = gm.GridModel(el, clust, pat_clr, eq)

    # solve overall objective fn - change to match run
    if isinstance(load_params, str):
        for K in range(nclust):
            pat_gm.k[K] = all_ks[K, pat_num]
    pat_gm.solve_overall_sse()
    grid_err = pat_gm.get_grid_sse()
    if isinstance(save_grid_sse, str):
        all_grid_sse[..., pat_num] = grid_err
        np.savez(save_grid_sse, all_grid_sse=all_grid_sse)
    ks = np.array([pe.value(pat_gm.k[K]) for K in range(pat_gm.nK)])
    if isinstance(save_params, str):
        all_ks[:, pat_num] = ks
        np.savez(save_params, all_ks=all_ks)
