#!/usr/bin/env python3
"""Plot the results from PeleC restarts
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
def parse_ic(fname):
    """Parse the file written by PeleC to understand the initial condition

    Returns a dictionary for easy acces
    """

    # Read into dataframe
    df = pd.read_csv(fname)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # convert to dictionary for easier access
    return df.to_dict("records")[0]


# ========================================================================
def load_datlog(fdir):
    """Load the data log and normalize """

    icname = "ic.txt"
    datname = "datlog"

    ics = parse_ic(os.path.join(fdir, icname))
    df = pd.read_csv(os.path.join(fdir, datname), delim_whitespace=True)

    # proper dimensions
    df["time_norm"] = df["time"] / ics["tau"]
    rho = 1.0 / (Omega) * df["mass"]
    KE0 = 3. / 2. * ics["urms0"] ** 2
    # KE0 = df["rho_K"].iloc[0] / (ics["rho0"] * Omega)
    df["rho_K_norm"] = 1.0 / (ics["rho0"] * Omega * KE0) * df["rho_K"]
    df["enstr_norm"] = (
        1.0 / (ics["rho0"] * Omega) * df["enstr"] * (ics["lambda0"] / ics["urms0"]) ** 2
    )

    return df, ics


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
        for d in ["case000012", "case000011", "case000010", "case000009", "case000008"]
    ]
    odir = os.path.abspath("./plots")
    if not os.path.exists(odir):
        os.makedirs(odir)
    fmt = "png"
    L = 2 * np.pi
    Omega = L ** 2
    pdirs = ["interp", "result"]
    terror = 1
    tplot = 5

    # Loop on directories
    elst = []
    for pdir in pdirs:
        # Plot reference data
        odf, ics = load_datlog(os.path.join(fdirs[0], "original"))
        plt.close("all")

        plt.figure(0)
        p = plt.plot(odf.time_norm, odf.enstr_norm, color=cmap[-1], lw=1, zorder=10)
        p[0].set_dashes(dashseq[0])

        plt.figure(1)
        p = plt.plot(odf.time_norm, odf.rho_K_norm, color=cmap[-1], lw=1, zorder=10)
        p[0].set_dashes(dashseq[0])

        for k, fdir in enumerate(fdirs):
            ppdir = os.path.join(fdir, pdir)
            print("Processing {0:s}".format(ppdir))

            # Get initial conditions
            df, _ = load_datlog(ppdir)
            df["rho_K_error"] = np.fabs(df.rho_K_norm - odf.rho_K_norm) / odf.rho_K_norm
            df["enstr_error"] = np.fabs(df.enstr_norm - odf.enstr_norm) / odf.rho_K_norm

            plt.figure(0)
            p = plt.plot(df.time_norm, df.enstr_norm, color=cmap[k], lw=2)
            p[0].set_dashes(dashseq[k])

            plt.figure(1)
            p = plt.plot(df.time_norm, df.rho_K_norm, color=cmap[k], lw=2)
            p[0].set_dashes(dashseq[k])

            idx = np.fabs(df.time_norm - terror).idxmin()
            stats = np.load(os.path.join(fdir, "statistics.npz"))
            elst.append(
                {
                    "pdir": pdir,
                    "lmask": stats["lmask"] / ics["lambda0"],
                    "KE": df.rho_K_error.iloc[idx],
                    "enstr": df.enstr_error.iloc[idx],
                }
            )

        # Format plots
        plt.figure(0)
        ax = plt.gca()
        plt.xlabel(r"$t / \tau$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\omega$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.xlim([0, tplot])
        plt.ylim([0, 0.8])
        plt.gcf().subplots_adjust(bottom=0.147)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "enstrophy_{0:s}.{1:s}".format(pdir, fmt)),
            format=fmt,
            dpi=300,
        )

        plt.figure(1)
        ax = plt.gca()
        plt.xlabel(r"$t / \tau$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$K$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.xlim([0, tplot])
        plt.ylim([0, 0.6])
        plt.gcf().subplots_adjust(bottom=0.147)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "KE_{0:s}.{1:s}".format(pdir, fmt)), format=fmt, dpi=300
        )

    # Get all the errors
    errors = pd.DataFrame(elst)

    plt.figure(2)
    for cnt, pdir in enumerate(pdirs):
        sdf = errors[errors.pdir == pdir]
        plt.plot(
            sdf.lmask,
            sdf.enstr,
            color=cmap[cnt],
            mec=cmap[cnt],
            mfc=cmap[cnt],
            marker=markertype[cnt],
            ms=10,
        )
    ax = plt.gca()
    plt.xlabel(r"$L_m / \lambda$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_\omega$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.147)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(
        os.path.join(odir, "enstrophy_error.{0:s}".format(fmt)), format=fmt, dpi=300
    )

    plt.figure(3)
    for cnt, pdir in enumerate(pdirs):
        sdf = errors[errors.pdir == pdir]
        plt.plot(
            sdf.lmask,
            sdf.KE,
            color=cmap[cnt],
            mec=cmap[cnt],
            mfc=cmap[cnt],
            marker=markertype[cnt],
            ms=10,
        )
    ax = plt.gca()
    plt.xlabel(r"$L_m / \lambda$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$e_K$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.147)
    plt.gcf().subplots_adjust(left=0.16)
    plt.savefig(os.path.join(odir, "KE_error.{0:s}".format(fmt)), format=fmt, dpi=300)

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
