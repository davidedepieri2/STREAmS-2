#!/usr/bin/env python3
"""
Reconstruct an approximate wall-normal profile of the SGS viscosity mu_sgs(y)
from an existing STREAmS postpro `.prof` file, without any solver-side change.

Rationale
---------
`postpro_bl.F90` writes, per x-station, a column `mum(y) = <mu_combined>(y) / <mu_combined>(y_wall)`
where mu_combined = mu_molecular + mu_sgs (STREAmS never stores mu_sgs alone).
Since the WALE model forces mu_sgs -> 0 at the wall, <mu_combined>(wall) ~= mu_molecular(T_wall).
So, using the mean-temperature column ttm(y) (already in the same file) and Sutherland's law,

    mu_sgs(y) / mu_wall  ~=  mum(y) - f(T(y)) / f(T_wall)

with f(T) = T**1.5 * (1+Sr) / (T+Sr), Sr = s_suth/T_ref (both read from the case's .ini).

This is an approximation: it uses f(<T>) instead of <f(T)> (Sutherland is nonlinear in T),
so it is meant for a qualitative check (e.g. does mu_sgs grow like y**3 near the wall,
as WALE's near-wall asymptotic behaviour predicts) rather than a quantitative one.
For an exact profile, mu_sgs must be accumulated as its own nv_stat slot in the solver.

Usage
-----
    python3 mu_sgs_yscaling.py <case_dir> <prof_file> [--out FILE.png]

<case_dir>   directory containing singleideal.ini (to read s_suth, T_ref)
<prof_file>  a POSTPRO/stat_XXXXXX.prof file produced by postpro_bl.F90
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Column layout written by postpro_bl.F90 (1-indexed in the Fortran comment, 0-indexed here)
COL_Y = 0
COL_TTM = 4
COL_MUM = 6
COL_YP = 8   # y+ (wall units)


def read_ini_param(ini_path: Path, key: str) -> float:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([0-9.eEdD+\-]+)", re.IGNORECASE)
    with open(ini_path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                token = m.group(1).replace("D", "E").replace("d", "E")
                return float(token)
    raise KeyError(f"'{key}' not found in {ini_path}")


def sutherland_f(t_nondim: np.ndarray, sr: float) -> np.ndarray:
    """f(T) = T^1.5 * (1+Sr)/(T+Sr), T already non-dimensional (T/T_ref)."""
    return t_nondim**1.5 * (1.0 + sr) / (t_nondim + sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir", type=Path, help="directory containing singleideal.ini")
    ap.add_argument("prof_file", type=Path, help="POSTPRO/stat_XXXXXX.prof file")
    ap.add_argument("--out", type=Path, default=None, help="output PNG (default: alongside prof_file)")
    args = ap.parse_args()

    ini_path = args.case_dir / "singleideal.ini"
    s_suth = read_ini_param(ini_path, "s_suth")
    t_ref = read_ini_param(ini_path, "T_ref")
    sr = s_suth / t_ref
    print(f"[info] s_suth={s_suth}, T_ref={t_ref}  ->  Sr=s_suth/T_ref={sr:.4f}")

    data = np.loadtxt(args.prof_file)
    y = data[:, COL_Y]
    ttm = data[:, COL_TTM]
    mum = data[:, COL_MUM]
    yp = data[:, COL_YP]

    f_t = sutherland_f(ttm, sr)
    f_twall = f_t[0]  # first grid point = wall
    mu_sgs_over_muw = mum - f_t / f_twall

    # Only the near-wall region is physically meaningful for a y^3 check;
    # far from the wall mu_sgs can be a sizeable fraction of mu, near the wall
    # it's a small residual of two close numbers -> can be noisy/slightly negative.
    valid = mu_sgs_over_muw > 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(y, mum, label=r"$\mu_{mol+sgs}/\mu_w$ (existing postpro output)")
    ax.plot(y[valid], mu_sgs_over_muw[valid], "o", ms=3, label=r"$\mu_{sgs}/\mu_w$ (reconstructed)")
    ax.set_xlabel("y")
    ax.set_ylabel(r"$\mu/\mu_w$")
    ax.legend(fontsize=8)
    ax.set_title("Combined vs. reconstructed SGS viscosity")

    ax = axes[1]
    ax.loglog(yp[valid], mu_sgs_over_muw[valid], "o", ms=4, label=r"$\mu_{sgs}/\mu_w$ (reconstructed)")
    if valid.sum() > 3:
        # reference y^3 slope anchored at the last near-wall valid point actually used
        i_anchor = np.where(valid)[0][min(5, valid.sum() - 1)]
        y3 = (yp / yp[i_anchor]) ** 3 * mu_sgs_over_muw[i_anchor]
        ax.loglog(yp[valid], y3[valid], "--", color="gray", label=r"$y^{+3}$ reference slope")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$\mu_{sgs}/\mu_w$")
    ax.legend(fontsize=8)
    ax.set_title("Near-wall scaling check (log-log)")

    fig.tight_layout()
    out = args.out or args.prof_file.with_suffix(".mu_sgs_check.png")
    fig.savefig(out, dpi=150)
    print(f"[info] wrote {out}")


if __name__ == "__main__":
    main()
