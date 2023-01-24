import os
import sys

import numpy as np

sys.path.append(os.path.abspath("../"))
import psanalysis.analyze_grids as ag

cwd = os.getcwd()
tpts = np.arange(0, 80, 2)
test_dir = ag.PlanarStudy(os.path.join(cwd, "SampleImages"), tpts)
test_dir.preprocess_scans(debug=True)
