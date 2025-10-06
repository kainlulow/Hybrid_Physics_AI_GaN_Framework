#!/usr/bin/env python
# coding: utf-8

# ---------- I/O helpers (unchanged) ----------
from __future__ import division
import os
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# (Optional sklearn/matplotlib imports kept if you use them elsewhere)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA

# ------------------ Existing parsers ------------------
def read_gtree(filename):
    simulation_flow = []
    simulation_tree = []
    with open(filename, "r") as f:
        line = f.readline()
        while line:
            if line.split():
                if line.split()[-1] == "flow":
                    line = f.readline()
                    while line:
                        simulation_flow.append([line.split()[0]+"_"+line.split()[1]])
                        line = f.readline()
                        if line.split()[-1] == "variables":
                            break
                if line.split()[-1] == "tree":
                    line = f.readline()
                    while line:
                        if not line.split()[3].split("{")[-1].split("}")[0]:
                            tmp_tree = "-"
                        else:
                            tmp_tree = line.split()[3].split("{")[-1].split("}")[0]
                        simulation_tree.append([line.split()[0],
                                                line.split()[1], line.split()[2],
                                                tmp_tree])
                        line = f.readline()
                        if not line.split():
                            break
            line = f.readline()
    numcol = len(simulation_flow) - 1
    row_N = [None] * len(simulation_flow)
    row_V = [None] * len(simulation_flow)
    Workbench_N, Workbench_V = [], []
    Variable_name = [elem for sublist in simulation_flow for elem in sublist]
    for i in range(len(simulation_tree)):
        if int(simulation_tree[i][0]) == numcol:
            row_N[int(simulation_tree[i][0])] = simulation_tree[i][1]
            row_V[int(simulation_tree[i][0])] = simulation_tree[i][-1]
            Workbench_N.append(row_N.copy()); Workbench_V.append(row_V.copy())
        else:
            row_N[int(simulation_tree[i][0])] = simulation_tree[i][1]
            row_V[int(simulation_tree[i][0])] = simulation_tree[i][-1]
    Workbench_N.insert(0, Variable_name); Workbench_V.insert(0, Variable_name)
    return Workbench_N, Workbench_V

def read_plt(filename):
    try:
        name_tmp, data_tmp, variable_name, data_result = [], [], [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line:
                if line.split():
                    if line.strip().split()[0] == "datasets":
                        while line:
                            line = f.readline()
                            for value in line.split():
                                if value.strip('"') != "]":
                                    name_tmp.append(value.strip('"'))
                            if line.strip().split()[-1] == "]":
                                break
                    if line.strip().split()[0] == "Data":
                        while line:
                            line = f.readline()
                            for value in line.split():
                                data_tmp.append(value.strip('"'))
                            if line.strip().split()[-1] == "}":
                                break
                line = f.readline()
        for i in range(len(name_tmp)):
            if i == 0:
                variable_name.append(name_tmp[i])
            if (i != 0) & (i % 2 == 0):
                variable_name.append(name_tmp[i - 1] + "_" + name_tmp[i])
        row = []
        for i in range(len(data_tmp[:-1])):
            row.append(float(data_tmp[i]))
            if len(row) == len(variable_name):
                data_result.append(row); row = []
        data_result.insert(0, variable_name)
        return data_result
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return []



# ============================================================
#                 NEW METRICS (this work)
# ============================================================
def _asc(vg, id_):
    """Return (vg,id_) sorted by vg ascending with NaNs removed."""
    vg = np.asarray(vg, float); id_ = np.asarray(id_, float)
    m = np.isfinite(vg) & np.isfinite(id_)
    vg, id_ = vg[m], id_[m]
    order = np.argsort(vg)
    return vg[order], id_[order]

def _find_v_at_current(vg, id_abs, itarget, v_pref=None):
    """Interpolate Vg where |Id|=itarget. If multiple segments cross, prefer
    the one closest to v_pref; otherwise pick the steepest in log-space."""
    vg, id_abs = _asc(vg, id_abs)
    diff = id_abs - itarget; sgn = np.sign(diff)
    idxs = np.where(sgn[:-1] * sgn[1:] <= 0)[0]
    if len(idxs) == 0:
        return np.nan
    cand = []
    for i in idxs:
        x0, x1 = vg[i], vg[i+1]; y0, y1 = id_abs[i], id_abs[i+1]
        vc = x0 if y1 == y0 else x0 + (itarget - y0) * (x1 - x0) / (y1 - y0)
        sc = abs((np.log10(y1) - np.log10(y0)) / (x1 - x0 + 1e-15)) if (y0>0 and y1>0) else 0.0
        cand.append((float(vc), sc))
    if v_pref is not None and np.isfinite(v_pref):
        return min((c[0] for c in cand), key=lambda v: abs(v - v_pref))
    return max(cand, key=lambda c: c[1])[0]

def find_vth_const_current(vg, id_abs, ith_abs):
    """Constant-current Vth at |Id|=ith_abs (steepest crossing)."""
    return _find_v_at_current(vg, id_abs, abs(ith_abs), v_pref=None)

def ss_two_point_mVdec(vg, id_abs, ith_abs, decades_low=1, decades_high=3):
    """
    Two-point SS between:
      I_high = Ith / 10^decades_low   (e.g., Ith/10)
      I_low  = Ith / 10^decades_high  (e.g., Ith/1000)
    SS = (Vg(I_high) - Vg(I_low)) / (decades_high - decades_low)  [V/dec] -> mV/dec
    """
    #if decades_high <= decades_low:
    #    raise ValueError("Require decades_high > decades_low (e.g., 3 > 1).")
    vg, id_abs = _asc(vg, id_abs)
    ith = abs(ith_abs)
    I_high = ith * (10.0 ** decades_low)
    I_low  = ith / (10.0 ** decades_high)
    vth_est = _find_v_at_current(vg, id_abs, ith)
    v_hi = _find_v_at_current(vg, id_abs, I_high, v_pref=vth_est)
    v_lo = _find_v_at_current(vg, id_abs, I_low,  v_pref=vth_est)
    if not (np.isfinite(v_hi) and np.isfinite(v_lo)):
        return np.nan
    dV = np.abs(v_hi - v_lo); ddec = float(decades_high + decades_low)
    return abs(dV / ddec) * 1e3

def extract_metrics_from_curve(
    Vgvec, Idvec,
    mostype='p',
    Ith_abs=1e-5,
    Vdd=None,
    overdrive_on_abs=5.0,
    overdrive_off_abs=2.0,
    Ioff_bias_V=0.0
):
    """
    Returns:
      {
        'Vth', 'Ion_eq', 'Ion_fixedV',
        'Ioff_fixed0', 'Ioff_eq',
        'SS_mVdec', 'Vov_on', 'Vov_off', 'Vdd'
      }
    """
    vg, id_ = _asc(Vgvec, Idvec)
    id_abs = np.abs(id_)
    ith = abs(Ith_abs)

    Vth = find_vth_const_current(vg, id_abs, ith)

    # sign helper: nMOS(+1), pMOS(-1)
    s = +1 if str(mostype).lower() == 'n' else -1

    # ON equal-overdrive
    Ion_eq = np.nan; Vov_on = s * float(overdrive_on_abs)
    if np.isfinite(Vth):
        v_target_on = Vth + Vov_on
        if vg.min() <= v_target_on <= vg.max():
            Ion_eq = float(np.interp(v_target_on, vg, id_abs))

    # ON at fixed VGS = Vdd
    Ion_fixedV = np.nan
    if Vdd is not None and (vg.min() <= Vdd <= vg.max()):
        Ion_fixedV = float(np.interp(float(Vdd), vg, id_abs))

    # OFF at fixed VGS = Ioff_bias_V (usually 0 V)
    Ioff_fixed0 = np.nan
    if vg.min() <= Ioff_bias_V <= vg.max():
        Ioff_fixed0 = float(np.interp(float(Ioff_bias_V), vg, id_abs))
    
    ON_OFF_fixedV = Ion_fixedV /Ioff_fixed0

    # OFF equal-overdrive (subthreshold side)
    Ioff_eq = np.nan; Vov_off = -s * float(overdrive_off_abs)
    if np.isfinite(Vth):
        v_target_off = Vth + Vov_off
        if vg.min() <= v_target_off <= vg.max():
            Ioff_eq = float(np.interp(v_target_off, vg, id_abs))
    
    ON_OFF_eq = Ion_eq/Ioff_eq


    # Two-point SS (Ith/10 .. Ith/1000)
    SS_mVdec = ss_two_point_mVdec(vg, id_abs, ith, decades_low=1, decades_high=3)

    return {
        'Vth': Vth,
        'Ion_eq': Ion_eq,
        'Ion_fixedV': Ion_fixedV,
        'Ioff_fixed0': Ioff_fixed0,
        'Ioff_eq': Ioff_eq,
        'ON_OFF_eq': ON_OFF_eq,
        'ON_OFF_fixedV': ON_OFF_fixedV,
        'SS_mVdec': SS_mVdec,
        'Vov_on': Vov_on,
        'Vov_off': Vov_off,
        'Vdd': Vdd
    }

def resample_iv_on_grid(Vgvec, Idvec, Vg_grid, use_abs=True):
    """Interpolate curve onto a common Vg grid (returns |Id| if use_abs)."""
    vg, id_ = _asc(Vgvec, Idvec)
    y = np.abs(id_) if use_abs else id_
    return np.interp(Vg_grid, vg, y)
