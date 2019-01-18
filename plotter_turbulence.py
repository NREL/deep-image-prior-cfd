#!/usr/bin/env python3
"""Plot the results
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt

# from scipy.stats import gaussian_kde

# ========================================================================
#
# Some defaults variables
#
# ========================================================================
plt.rc("text", usetex=True)
cmap_med = [
    "#F15A60",
    "#7AC36A",
    "#5A9BD4",
    "#FAA75B",
    "#9E67AB",
    "#CE7058",
    "#D77FB4",
    "#737373",
]
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [3, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h"]

# ========================================================================
#
# Functions
#
# ========================================================================


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Setup
    rdir = os.path.abspath("turbulence/results")
    fdirs = [
        os.path.join(rdir, d)
        for d in os.listdir(rdir)
        if os.path.isdir(os.path.join(rdir, d))
    ]
    odir = os.path.abspath("./plots")
    if not os.path.exists(odir):
        os.makedirs(odir)
    stats = pd.read_csv(os.path.join(rdir, "statistics.dat"))
    stats["error_lambdar"] = np.fabs(stats.lambda0 - stats.lambdar) / stats.lambda0
    stats["error_urmsr"] = np.fabs(stats.urms0 - stats.urmsr) / stats.urms0
    stats["error_Sr"] = np.fabs(stats.S0 - stats.Sr) / stats.S0
    stats["error_lambdai"] = np.fabs(stats.lambda0 - stats.lambdai) / stats.lambda0
    stats["error_urmsi"] = np.fabs(stats.urms0 - stats.urmsi) / stats.urms0
    stats["error_Si"] = np.fabs(stats.S0 - stats.Si) / stats.S0
    _, f_colors = np.unique(stats.fraction, return_inverse=True)
    _, lm_colors = np.unique(stats.lmask, return_inverse=True)
    spectra = pd.read_csv(os.path.join(rdir, "spectra.dat"))
    spectra["error_r"] = np.fabs(spectra.Ek0 - spectra.Ekr) / spectra.Ek0
    spectra["error_i"] = np.fabs(spectra.Ek0 - spectra.Eki) / spectra.Ek0
    structure = pd.read_csv(os.path.join(rdir, "structure.dat"))
    L = 2 * np.pi
    fmt = "png"

    # ========================================================================
    # Show all groups on same scatter plots

    print("Mean lambda error DL: {0:f}".format(np.mean(stats.error_lambdar)))
    print("                  GP: {0:f}".format(np.mean(stats.error_lambdai)))
    print("Mean urms error DL: {0:f}".format(np.mean(stats.error_urmsr)))
    print("                GP: {0:f}".format(np.mean(stats.error_urmsi)))

    # Taylor microscale
    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    # plt.plot(stats.lambda0, stats.lambdar, "o", mec=cmap[0], mfc=cmap[0])
    plt.scatter(stats.lambda0, stats.lambdar, c=f_colors, alpha=0.3, cmap="Dark2_r")
    line = [np.min(stats.lambda0), np.max(stats.lambda0)]
    plt.plot(line, line, color=cmap[-1], lw=1)
    plt.xlabel(r"$\lambda_0$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$\lambda_r$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "lambda.{0:s}".format(fmt)), format=fmt, dpi=300)

    # urms
    plt.figure(1)
    plt.clf()
    ax = plt.gca()
    # plt.plot(stats.urms0, stats.urmsr, "o", mec=cmap[0], mfc=cmap[0])
    plt.scatter(stats.urms0, stats.urmsr, c=f_colors, alpha=0.3, cmap="Dark2_r")
    line = [np.min(stats.urms0), np.max(stats.urms0)]
    plt.plot(line, line, color=cmap[-1], lw=1)
    plt.xlabel(r"$u'_0$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$u'_r$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "urms.{0:s}".format(fmt)), format=fmt, dpi=300)

    # Skewness
    plt.figure(2)
    plt.clf()
    ax = plt.gca()
    plt.plot(stats.S0, stats.Sr, "o", mec=cmap[0], mfc=cmap[0])
    line = [np.min(stats.S0), np.max(stats.S0)]
    plt.plot(line, line, color=cmap[-1], lw=1)
    plt.xlabel(r"$S_0$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$S_r$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "skewness.{0:s}".format(fmt)), format=fmt, dpi=300)

    # Plot errors as a function of fraction
    mean_error = stats.groupby(["fraction"]).mean()

    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    plt.plot(mean_error.index, mean_error.error_lambdar, "o", mec=cmap[0], mfc=cmap[0])
    plt.xlabel(r"$f$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_\lambda$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(
        os.path.join(odir, "error_lambda_f.{0:s}".format(fmt)), format=fmt, dpi=300
    )

    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    plt.plot(mean_error.index, mean_error.error_urmsr, "o", mec=cmap[0], mfc=cmap[0])
    plt.xlabel(r"$f$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_{u'}$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(
        os.path.join(odir, "error_urms_f.{0:s}".format(fmt)), format=fmt, dpi=300
    )

    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    plt.plot(mean_error.index, mean_error.error_Sr, "o", mec=cmap[0], mfc=cmap[0])
    plt.xlabel(r"$f$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_S$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "error_S_f.{0:s}".format(fmt)), format=fmt, dpi=300)

    # Plot errors as a function of mask length scale
    mean_error = stats.groupby(["lmask"]).mean()

    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    plt.plot(mean_error.index, mean_error.error_lambdar, "o", mec=cmap[0], mfc=cmap[0])
    plt.xlabel(r"$L_m$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_\lambda$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(
        os.path.join(odir, "error_lambda_l.{0:s}".format(fmt)), format=fmt, dpi=300
    )

    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    plt.plot(mean_error.index, mean_error.error_urmsr, "o", mec=cmap[0], mfc=cmap[0])
    plt.xlabel(r"$L_m$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_{u'}$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(
        os.path.join(odir, "error_urms_l.{0:s}".format(fmt)), format=fmt, dpi=300
    )

    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    plt.plot(mean_error.index, mean_error.error_Sr, "o", mec=cmap[0], mfc=cmap[0])
    plt.xlabel(r"$L_m$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_S$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "error_S_l.{0:s}".format(fmt)), format=fmt, dpi=300)

    # Plot errors as a function of mask length scale for each fraction level
    mean_error = stats.groupby(["fraction", "lmask"]).mean()
    plt.close("all")
    for k, (lmask, df) in enumerate(mean_error.groupby(level=0)):

        plt.figure(0)
        y = df.error_lambdar
        x = df.index.levels[1][: len(y)] / (2 * np.pi)
        plt.plot(
            x, y, color=cmap[k], mec=cmap[k], mfc=cmap[k], marker=markertype[k], ms=10
        )
        y = df.error_lambdai
        plt.plot(
            x, y, color=cmap[k], mec=cmap[k], mfc="none", marker=markertype[k], ms=10
        )

        plt.figure(1)
        y = df.error_urmsr
        x = df.index.levels[1][: len(y)] / (2 * np.pi)
        plt.plot(
            x, y, color=cmap[k], mec=cmap[k], mfc=cmap[k], marker=markertype[k], ms=10
        )

        plt.figure(2)
        y = df.error_Sr
        x = df.index.levels[1][: len(y)] / (2 * np.pi)
        plt.plot(
            x, y, color=cmap[k], mec=cmap[k], mfc=cmap[k], marker=markertype[k], ms=10
        )

        plt.figure(3)
        y = df.ur_error
        x = df.index.levels[1][: len(y)] / (2 * np.pi)
        plt.plot(
            x, y, color=cmap[k], mec=cmap[k], mfc=cmap[k], marker=markertype[k], ms=10
        )

    plt.figure(0)
    ax = plt.gca()
    plt.xlabel(r"$L_m / L$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_\lambda$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(
        os.path.join(odir, "error_lambda.{0:s}".format(fmt)), format=fmt, dpi=300
    )

    plt.figure(1)
    ax = plt.gca()
    plt.xlabel(r"$L_m / L$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_{u'}$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "error_urms.{0:s}".format(fmt)), format=fmt, dpi=300)

    plt.figure(2)
    ax = plt.gca()
    plt.xlabel(r"$L_m / L$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_S$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "error_S.{0:s}".format(fmt)), format=fmt, dpi=300)

    plt.figure(3)
    ax = plt.gca()
    plt.xlabel(r"$L_m / L$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_u$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "error_u.{0:s}".format(fmt)), format=fmt, dpi=300)

    # ========================================================================
    # Loop on groups for single valued data
    grouped = stats.groupby(["fraction", "lmask"])
    for k, (name, group) in enumerate(grouped):

        # Taylor microscale
        plt.figure(0)
        plt.clf()
        ax = plt.gca()
        plt.plot(group.lambda0, group.lambdar, "o", mec=cmap[0], mfc=cmap[0])
        line = [np.min(stats.lambda0), np.max(stats.lambda0)]
        plt.plot(line, line, color=cmap[-1], lw=1)
        plt.xlabel(r"$\lambda_0$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\lambda_r$", fontsize=22, fontweight="bold")
        plt.title(r"$f={0:.3f}$ and $L_m={1:.3f}$".format(name[0], name[1] / L * 100))
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.14)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "lambda_{0:d}.{1:s}".format(k, fmt)), format=fmt, dpi=300
        )

        # urms
        plt.figure(1)
        plt.clf()
        ax = plt.gca()
        plt.plot(group.urms0, group.urmsr, "o", mec=cmap[0], mfc=cmap[0])
        line = [np.min(stats.urms0), np.max(stats.urms0)]
        plt.plot(line, line, color=cmap[-1], lw=1)
        plt.xlabel(r"$u'_0$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$u'_r$", fontsize=22, fontweight="bold")
        plt.title(r"$f={0:.3f}$ and $L_m={1:.3f}$".format(name[0], name[1] / L * 100))
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.14)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "urms_{0:d}.{1:s}".format(k, fmt)), format=fmt, dpi=300
        )

        # Skewness
        plt.figure(2)
        plt.clf()
        ax = plt.gca()
        plt.plot(group.S0, group.Sr, "o", mec=cmap[0], mfc=cmap[0])
        line = [np.min(stats.S0), np.max(stats.S0)]
        plt.plot(line, line, color=cmap[-1], lw=1)
        plt.xlabel(r"$S_0$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$S_r$", fontsize=22, fontweight="bold")
        plt.title(r"$f={0:.3f}$ and $L_m={1:.3f}$".format(name[0], name[1] / L * 100))
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.14)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "skewness_{0:d}.{1:s}".format(k, fmt)),
            format=fmt,
            dpi=300,
        )

    # ========================================================================
    # Spectra
    mean_spectra = spectra.groupby(["kbins"]).mean()
    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    p = plt.loglog(mean_spectra.index, mean_spectra.Ek0, color=cmap[0], lw=2)
    p[0].set_dashes(dashseq[0])
    p = plt.loglog(mean_spectra.index, mean_spectra.Ekr, color=cmap[1], lw=2)
    p[0].set_dashes(dashseq[1])
    p = plt.loglog(mean_spectra.index, mean_spectra.Eki, color=cmap[2], lw=2)
    p[0].set_dashes(dashseq[2])
    plt.xlabel(r"$k$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$E_k$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "spectra.{0:s}".format(fmt)), format=fmt, dpi=300)

    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    p = plt.semilogx(mean_spectra.index, mean_spectra.error_r, color=cmap[1], lw=2)
    p[0].set_dashes(dashseq[1])
    p = plt.semilogx(mean_spectra.index, mean_spectra.error_i, color=cmap[2], lw=2)
    p[0].set_dashes(dashseq[2])
    plt.xlabel(r"$k$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_{E_k}$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(
        os.path.join(odir, "error_spectra.{0:s}".format(fmt)), format=fmt, dpi=300
    )

    # Get average within groups
    grouped = spectra.groupby(["fraction", "lmask"])
    for k, (name, group) in enumerate(grouped):

        mean_group = group.groupby(["kbins"]).mean()
        plt.figure(0)
        plt.clf()
        ax = plt.gca()
        p = plt.loglog(mean_group.index, mean_group.Ek0, color=cmap[0], lw=2)
        p[0].set_dashes(dashseq[0])
        p = plt.loglog(mean_group.index, mean_group.Ekr, color=cmap[1], lw=2)
        p[0].set_dashes(dashseq[1])
        p = plt.loglog(mean_group.index, mean_group.Eki, color=cmap[2], lw=2)
        p[0].set_dashes(dashseq[2])
        plt.xlabel(r"$k$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$E_k$", fontsize=22, fontweight="bold")
        plt.title(r"$f={0:.3f}$ and $L_m={1:.3f}$".format(name[0], name[1] / L * 100))
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.14)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "spectra_{0:d}.{1:s}".format(k, fmt)),
            format=fmt,
            dpi=300,
        )

    # ========================================================================
    # Structure functions
    mu = 0.25
    structure["theory_K41"] = structure.sf_orders / 3.0
    structure["theory_K62"] = (
        structure.sf_orders / 3.0 * (1 - mu * (structure.sf_orders - 3) / 6.0)
    )
    mean_structure = structure.groupby(["sf_orders"]).mean()

    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    p = plt.plot(
        mean_structure.index,
        mean_structure.zetas0,
        "o",
        marker=markertype[0],
        mec=cmap[0],
        mfc=cmap[0],
        ms=10,
    )
    p = plt.plot(
        mean_structure.index,
        mean_structure.zetasr,
        "o",
        marker=markertype[1],
        mec=cmap[1],
        mfc=cmap[1],
        ms=10,
    )
    p = plt.plot(
        mean_structure.index,
        mean_structure.zetasi,
        "o",
        marker=markertype[2],
        mec=cmap[2],
        mfc=cmap[2],
        ms=10,
    )
    p = plt.plot(mean_structure.index, mean_structure.theory_K41, color=cmap[-1], lw=1)
    p[0].set_dashes(dashseq[0])
    p = plt.plot(mean_structure.index, mean_structure.theory_K62, color=cmap[-1], lw=1)
    p[0].set_dashes(dashseq[1])
    plt.xlabel(r"$n$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$\zeta_n$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "structure.{0:s}".format(fmt)), format=fmt, dpi=300)

    # Get average within groups
    grouped = structure.groupby(["fraction", "lmask"])
    for k, (name, group) in enumerate(grouped):

        mean_group = group.groupby(["sf_orders"]).mean()
        plt.figure(0)
        plt.clf()
        ax = plt.gca()
        p = plt.plot(
            mean_group.index,
            mean_group.zetas0,
            "o",
            marker=markertype[0],
            mec=cmap[0],
            mfc=cmap[0],
            ms=10,
        )
        p = plt.plot(
            mean_group.index,
            mean_group.zetasr,
            "o",
            marker=markertype[1],
            mec=cmap[1],
            mfc=cmap[1],
            ms=10,
        )
        p = plt.plot(mean_group.index, mean_structure.theory_K41, color=cmap[-1], lw=1)
        p[0].set_dashes(dashseq[0])
        p = plt.plot(mean_group.index, mean_structure.theory_K62, color=cmap[-1], lw=1)
        p[0].set_dashes(dashseq[1])
        plt.xlabel(r"$n$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\zeta_n$", fontsize=22, fontweight="bold")
        plt.title(r"$f={0:.3f}$ and $L_m={1:.3f}$".format(name[0], name[1] / L * 100))
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.14)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "structure_{0:d}.{1:s}".format(k, fmt)),
            format=fmt,
            dpi=300,
        )

    # # ========================================================================
    # # Normalized longitudinal velocity gradient
    # mean_Zs = np.load(os.path.join(rdir, "mean_Zs.npz"))
    # Z0 = mean_Zs["Z0"]
    # Zr = mean_Zs["Zi"]
    # Zi = mean_Zs["Zr"]

    # pdf_space = np.linspace(-0.2, 0.2, 100)
    # pdf_Z0 = gaussian_kde(Z0.flatten())
    # pdf_Zr = gaussian_kde(Zr.flatten())
    # pdf_Zi = gaussian_kde(Zi.flatten())

    # plt.figure()
    # plt.semilogy(pdf_space, pdf_Z0(pdf_space), "-r")
    # plt.semilogy(pdf_space, pdf_Zr(pdf_space), "-g")
    # plt.semilogy(pdf_space, pdf_Zi(pdf_space), "-b")
    # plt.show()

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
