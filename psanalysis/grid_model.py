import os
import sys
from math import log

import numpy as np
import pyomo.dae as pdae
import pyomo.environ as pe
from plepy import PLEpy
from scipy.io import loadmat


def sflag(results):
    # Return flag corresponding to solver status
    # 0 = converged to optimal solution
    # 1 = problem may be infeasible
    # 2 = reached maximum number of iterations
    # 3 = sometimes there are other weird errors
    from pyomo.opt import SolverStatus, TerminationCondition

    stat = results.solver.status
    tcond = results.solver.termination_condition
    if ((stat == SolverStatus.ok) and
        (tcond == TerminationCondition.optimal)):
        flag = 0
    elif (tcond == TerminationCondition.infeasible):
        flag = 1
    elif (tcond == TerminationCondition.maxIterations):
        flag = 2
    else:
        flag = 3
    return flag


def get_bool_maps(el, diags=True, equal_flow=True, eq=None):
    """Function for defining flow constraints

    Parameters:
    el (ndarray): Elevation map

    Keywords:
    diags (bool): Whether or not to allow flow between diagonal grids,
        True by default
    equal_flow (bool): If true, magnitude of flow is equal in all possible
        directions. Else, flow is proportional to difference in elevation.
        True by default
    eq (bool type ndarray): ND array defining where to allow flow at equal
        elevation. 1=can flow to equal elevations, 0=can only flow downhill

    Returns:
    List of boolean arrays corresponding to whether the grids can flow in each
    cardinal direction. Order of directions: N, NE, E, SE, S, SW, W, NW
    """

    def get_dir_bool(yslice, xslice):
        # flow based on elevation
        el_part = el_bool[1:-1, 1:-1] - el_bool[yslice, xslice]
        # whether to allow flow at equal elevation
        eq_part = np.logical_and(eq_bool[1:-1, 1:-1], eq_bool[yslice, xslice])
        eq_part = eq_part.astype("int")
        combined = (el_part + eq_part)/(ndir/2)
        return combined

    # set map properties
    ny, nx = np.shape(el)
    if not isinstance(eq, np.ndarray):
        eq = np.ones_like(el)
    eq_bool = np.zeros((ny+2, nx+2))
    eq_bool[1:-1, 1:-1] = eq
    if diags:
        ndir = 8
    else:
        ndir = 4

    # preprocess elevation map
    out_lung = el == 0
    barrier = int(np.max(el)+1)
    el_bool = (barrier+1)*np.ones(((ny+2), (nx+2)))
    el[out_lung] = barrier
    exit_blocks = np.vstack(
        (
            np.zeros((1, nx+2), dtype=bool),
            np.hstack((el == 1, np.zeros((ny, 2), dtype=bool))),
            np.zeros((1, nx+2), dtype=bool)
        )
    )
    el_bool[exit_blocks] = 0
    el_bool[1:-1, 1:-1] = el

    # subtract elevation from specified direction
    # add 1 if allowing flow between grids of equal elevation
    # denominator is used when flow is proportional rather than equal
    bool_n = get_dir_bool(slice(0, -2), slice(1, -1))
    bool_ne = get_dir_bool(slice(0, -2), slice(2, None))
    bool_e = get_dir_bool(slice(1, -1), slice(2, None))
    bool_se = get_dir_bool(slice(2, None), slice(2, None))
    bool_s = get_dir_bool(slice(2, None), slice(1, -1))
    bool_sw = get_dir_bool(slice(2, None), slice(0, -2))
    bool_w = get_dir_bool(slice(1, -1), slice(0, -2))
    bool_nw = get_dir_bool(slice(0, -2), slice(0, -2))

    # remove negative values
    bool_n[bool_n <= 0] = 0
    bool_ne[bool_ne <= 0] = 0
    bool_e[bool_e <= 0] = 0
    bool_se[bool_se <= 0] = 0
    bool_s[bool_s <= 0] = 0
    bool_sw[bool_sw <= 0] = 0
    bool_w[bool_w <= 0] = 0
    bool_nw[bool_nw <= 0] = 0
    if equal_flow:
        bool_n = bool_n > 0
        bool_ne = bool_ne > 0
        bool_e = bool_e > 0
        bool_se = bool_se > 0
        bool_s = bool_s > 0
        bool_sw = bool_sw > 0
        bool_w = bool_w > 0
        bool_nw = bool_nw > 0
    if not diags:
        bool_ne = np.zeros((ny, nx))
        bool_se = np.zeros((ny, nx))
        bool_nw = np.zeros((ny, nx))
        bool_sw = np.zeros((ny, nx))
    # make numerical
    bool_n = bool_n.astype("float")
    bool_ne = bool_ne.astype("float")
    bool_nw = bool_nw.astype("float")
    bool_s = bool_s.astype("float")
    bool_se = bool_se.astype("float")
    bool_sw = bool_sw.astype("float")
    bool_e = bool_e.astype("float")
    bool_w = bool_w.astype("float")
    return [bool_n, bool_ne, bool_e, bool_se, bool_s, bool_sw, bool_w, bool_nw]


class GridModel(pe.ConcreteModel):
    # Class defining PYOMO model for specified grid model

    ts = list(range(0, 80, 2))
    ts8 = list(range(8, 80, 2))

    def __init__(self, el, clust, fixlow, cdata, eq, wmat=None):
        """Define Concrete PYOMO Model for model & data specified

        Parameters:
        el (ndarray): Elevation map
        clust (ndarray): ND array defining which cluster each grid belongs to.
            To fit a parameter for each grid, specify unique cluster values for
            each grid in lung
        fixlow (ndarray): ND boolean array defining grids to fix the clearance
            rate parameter for
        cdata (ndarray): Concentration data of shape (ny, nx, nt)
        eq (ndarray): ND boolean array defining where to allow flow at equal
            elevation. 1=can flow to equal elevations, 0=can only flow downhill

        Keywords:
        wmat (ndarray): Weight matrix for each grid in overall objective. If
            None, equal weights will be used. None by default.

        Returns:
        Instance of class GridModel

        Methods:
        solve_grids: Use to solve for ideal solution for each grid (see paper
            for details). Must be done *prior* to fitting overall objective.
        solve_overall_sse: Minimize overall objective function
        solve_cluster_sse: Minimize objective within an individual cluster,
            neglecting error in other clusters
        """
        pe.ConcreteModel.__init__(self)
        # generic attributes
        self.ny, self.nx = np.shape(el)
        self.ngrid = self.ny*self.nx
        self.clust = clust
        self.nK = np.max(self.clust)
        self.fixlow = fixlow
        self.nfix = np.sum(self.fixlow > 0)
        self.out_lung = el == 0
        self.bool_map = get_bool_maps(el, eq=eq)
        self.sum_leave = np.clip(np.sum(self.bool_map, axis=0), 1, 8)
        self.bool_n = self.bool_map[0]
        self.bool_ne = self.bool_map[1]
        self.bool_e = self.bool_map[2]
        self.bool_se = self.bool_map[3]
        self.bool_s = self.bool_map[4]
        self.bool_sw = self.bool_map[5]
        self.bool_w = self.bool_map[6]
        self.bool_nw = self.bool_map[7]

        # PYOMO attributes
        # indices
        self.t = pdae.ContinuousSet(bounds=(8, 80), initialize=self.ts8)
        self.i = pe.RangeSet(0, self.ny-1)
        self.j = pe.RangeSet(0, self.nx-1)
        self.K = pe.RangeSet(0, self.nK)
        # fitted parameters
        self.k = pe.Var(self.K, initialize=1.0, bounds=(1e-3, 1e3))
        self.k[self.nK] = 1e-2
        self.k[self.nK].fix()
        fixclust = self.clust[self.fixlow > 0]
        fixmed = self.fixlow[self.fixlow > 0]
        for i in range(self.nfix):
            self.k[fixclust[i]] = fixmed[i]
            self.k[fixclust[i]].fix()
        # states
        self.C = pe.Var(self.t, self.i, self.j, within=pe.NonNegativeReals)
        # derivatives
        self.dCdt = pdae.DerivativeVar(self.C, wrt=self.t)

        # data attributes
        if wmat is None:
            self.wmat = np.ones((self.ny, self.nx))
        else:
            emsg = f"Weight matrix dimensions should be ({self.ny},{self.nx})."
            assert np.shape(wmat) == (self.ny, self.nx), emsg
            self.wmat = wmat
        self.cdata = cdata
        self.C8 = np.clip(np.mean(self.cdata[..., 3:6], axis=2), 1e-3, 100)
        for t in self.t:
            for i in self.i:
                for j in self.j:
                    self.C[t, i, j] = self.C8[i, j]
                    self.dCdt[t, i, j] = -1e-2*self.C8[i, j]

    def _init_cond(self, m, i, j):
        if self.out_lung[i, j]:
            return m.C[8, i, j] == 0.
        else:
            return m.C[8, i, j] == self.C8[i, j]

    def _odes(self, m, t, i, j):
        if self.out_lung[i, j]:
            return m.dCdt[t, i, j] == 0.
        else:
            K = self.clust[i, j]
            rhs = 0.
            rhs = rhs - m.k[K] * m.C[t, i, j]
            if i < self.ny-1:
                rhs = rhs + (self.bool_n[i+1, j] * m.k[self.clust[i+1, j]]
                             * m.C[t, i+1, j]) / self.sum_leave[i+1, j]
            if (i < self.ny-1) and (j > 0):
                rhs = rhs + (self.bool_ne[i+1, j-1] * m.k[self.clust[i+1, j-1]]
                             * m.C[t, i+1, j-1]) / self.sum_leave[i+1, j-1]
            if j > 0:
                rhs = rhs + (self.bool_e[i, j-1] * m.k[self.clust[i, j-1]]
                             * m.C[t, i, j-1]) / self.sum_leave[i, j-1]
            if (j > 0) and (i > 0):
                rhs = rhs + (self.bool_se[i-1, j-1] * m.k[self.clust[i-1, j-1]]
                             * m.C[t, i-1, j-1]) / self.sum_leave[i-1, j-1]
            if i > 0:
                rhs = rhs + (self.bool_s[i-1, j] * m.k[self.clust[i-1, j]]
                             * m.C[t, i-1, j]) / self.sum_leave[i-1, j]
            if (i > 0) and (j < self.nx-1):
                rhs = rhs + (self.bool_sw[i-1, j+1] * m.k[self.clust[i-1, j+1]]
                             * m.C[t, i-1, j+1]) / self.sum_leave[i-1, j+1]
            if j < self.nx-1:
                rhs = rhs + (self.bool_w[i, j+1] * m.k[self.clust[i, j+1]]
                             * m.C[t, i, j+1]) / self.sum_leave[i, j+1]
            if (j < self.nx-1) and (i < self.ny-1):
                rhs = rhs + (self.bool_nw[i+1, j+1] * m.k[self.clust[i+1, j+1]]
                             * m.C[t, i+1, j+1]) / self.sum_leave[i+1, j+1]
            return m.dCdt[t, i, j] == 0.1*rhs

    def _grid_obj(self, m):
        i, j = self.gid
        sse_ij = sum([self.wmat[i, j]*(m.C[self.ts[t], i, j]
                                       - self.cdata[i, j, t])**2
                      for t in range(4, 40)])
        return sse_ij

    def _overall_obj_sse(self, m):
        sse = 0.
        for cl in range(self.nK):
            gids = np.c_[np.where(self.clust == cl)]
            for gid in gids:
                i, j = gid
                sse_ij = sum([(m.C[self.ts[t], i, j] - self.cdata[i, j, t])**2
                              for t in range(4, 40)])
                sse += self.wmat[i,j] * (sse_ij - self.best_sse[i, j])
        return sse

    def _clust_obj(self, m):
        sse = 0.
        for gid in self.gids:
            i, j = gid
            sse_ij = sum([(m.C[self.ts[t], i, j] - self.cdata[i, j, t])**2
                          for t in range(4, 40)])
            sse += self.wmat[i,j] * (sse_ij - self.best_sse[i, j])
        return sse

    def solve_grids(self, solver=None, find_best=True, find_worst=False):
        # Method to find best fit when only considering each grid individually
        # (i.e. error in other grids can be humongous and we don't care)
        emsg = "Neither best nor worst fit selected."
        assert find_best or find_worst, emsg

        # activate constraints
        self.init_cond = pe.Constraint(self.i, self.j, rule=self._init_cond)
        self.odes = pe.Constraint(self.t, self.i, self.j, rule=self._odes)
        # define solver
        if solver is not None:
            self.solver = solver
        else:
            self.solver = pe.SolverFactory('ipopt')
            self.solver.options['linear_solver'] = 'ma97'
            self.solver.options['tol'] = 1e-6
            self.solver.options['max_iter'] = 600
            self.solver.options['acceptable_tol'] = 1e-4
        # numerical discretization
        tfd = pe.TransformationFactory("dae.finite_difference")
        tfd.apply_to(self, nfe=2*len(self.t), wrt=self.t, scheme="BACKWARD")

        # find best and/or worst SSE for each grid
        if find_best:
            self.best_sse = np.zeros((self.ny, self.nx))
        if find_worst:
            self.worst_sse = np.zeros((self.ny, self.nx))
        for cl in range(self.nK):
            self.gid = np.c_[np.where(self.clust == cl)][0]
            if find_best:
            # minimize objective
                self.grid_obj = pe.Objective(rule=self._grid_obj)
                results = self.solver.solve(self, keepfiles=False, tee=False,
                                            load_solutions=False)
                flag = sflag(results)
                if flag == 0:
                    self.solutions.load_from(results)
                    best_sse = pe.value(self.grid_obj)
                    self.best_sse[self.gid[0], self.gid[1]] = best_sse
                self.del_component(self.grid_obj)
            if find_worst:
                # maximize objective
                self.grid_obj = pe.Objective(rule=self._grid_obj,
                                            sense=pe.maximize)
                results = self.solver.solve(self, keepfiles=False, tee=False,
                                            load_solutions=False)
                flag = sflag(results)
                if flag == 0:
                    self.solutions.load_from(results)
                    worst_sse = pe.value(self.grid_obj)
                    self.worst_sse[self.gid[0], self.gid[1]] = worst_sse
                self.del_component(self.grid_obj)

    def solve_overall_sse(self, presolved=False):
        # Minimize overall objective function (sum of SSE between model & data
        # minus the best possible error for each grid) by varying all grid
        # clearance parameters. See paper for mathematical details.
        if presolved is False:
            # activate constraints
            self.init_cond = pe.Constraint(self.i, self.j,
                                           rule=self._init_cond)
            self.odes = pe.Constraint(self.t, self.i, self.j, rule=self._odes)
            # define solver
            if not hasattr(self, "solver"):
                self.solver = pe.SolverFactory('ipopt')
                self.solver.options['linear_solver'] = 'ma97'
                self.solver.options['tol'] = 1e-6
                self.solver.options['max_iter'] = 600
                self.solver.options['acceptable_tol'] = 1e-4
            # numerical discretization
            tfd = pe.TransformationFactory("dae.finite_difference")
            tfd.apply_to(self, nfe=2*len(self.t), wrt=self.t,
                         scheme="BACKWARD")
        self.overall_obj = pe.Objective(rule=self._overall_obj_sse)
        results = self.solver.solve(self, keepfiles=False, tee=True,
                                    load_solutions=False)
        flag = sflag(results)
        if flag == 0:
            self.solutions.load_from(results)

    def solve_cluster_sse(self, presolved=False):
        # Minimize error for a specific cluster, without regard to error in
        # other clusters. Similar to solve_grids, but for whole cluster.
        if presolved is False:
            # activate constraints
            self.init_cond = pe.Constraint(self.i, self.j,
                                           rule=self._init_cond)
            self.odes = pe.Constraint(self.t, self.i, self.j, rule=self._odes)
            # define solver
            if not hasattr(self, "solver"):
                self.solver = pe.SolverFactory('ipopt')
                self.solver.options['linear_solver'] = 'ma97'
                self.solver.options['tol'] = 1e-6
                self.solver.options['max_iter'] = 600
                self.solver.options['acceptable_tol'] = 1e-4
            # numerical discretization
            tfd = pe.TransformationFactory("dae.finite_difference")
            tfd.apply_to(self, nfe=2*len(self.t), wrt=self.t,
                         scheme="BACKWARD")
        self.obj = pe.Objective(rule=self._clust_obj)
        results = self.solver.solve(self, keepfiles=False, tee=True,
                                    load_solutions=False)
        flag = sflag(results)
        if flag == 0:
            self.solutions.load_from(results)

    def get_grid_sse(self):
        # Calculate error in each grid from optimized solution
        self.solved_error = np.zeros((self.ny, self.nx))
        for i in range(ny):
            for j in range(nx):
                sse_ij = sum([(pe.value(self.C[self.ts[t], i, j])
                               - self.cdata[i, j, t])**2
                              for t in range(4, 40)])
                self.solved_error[i, j] = sse_ij
        return self.solved_error
