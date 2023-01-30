import os
import sys

import numpy as np

sys.path.append(os.path.abspath("../"))
import psanalysis.analyze_grids as ag

wd = os.path.dirname(os.path.realpath(__file__))
tpts = np.arange(0, 80, 2)
test_dir = ag.PlanarStudy(os.path.join(wd, "SampleImages", "Subject1"), tpts)
test_dir.preprocess_scans(debug=False)
(a, grid_aTc, grid_aIn, grid_pTc, grid_pIn) = test_dir.gridify(16, 8)
