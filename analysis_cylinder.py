#!/usr/bin/env python3
"""Run through all the results and perform turbulence analysis on each
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import argparse
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from mpl_toolkits import axes_grid1
from mpi4py import MPI

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
def get_scaling(logname):
    with open(logname, "r") as f:
        for line in f:
            if "Image name" in line:
                imgname = line.split()[-1]
                imgdir = os.path.dirname(imgname)
                base = os.path.splitext(os.path.basename(imgname))[0]

    fpath = os.path.dirname(os.path.realpath(__file__))
    imgdir = fpath + imgdir.split(os.path.basename(fpath), 1)[1]
    sname = os.path.join(imgdir, "scalings.dat")
    df = pd.read_csv(sname)

    sdf = df[df.filename == base]
    return (
        [np.float(sdf.ux_lbnd), np.float(sdf.uz_lbnd), 0.0],
        [np.float(sdf.ux_rbnd), np.float(sdf.uz_rbnd), 0.0],
    )


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.

   Taken from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
   """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


# ========================================================================
def make_velocity_plots(fdir, u, mask, sfx):
    """Make some plots of velocity and velocity magnitude"""

    names = ["u", "v", "umag"]
    nlevels = 10
    min_plt = [-0.4, -1]
    max_plt = [1.5, 1]
    xbnd = [-2.0, 15.0]
    ybnd = [-4.258317025440313, 4.258317025440313]
    extent = [xbnd[0], xbnd[1], ybnd[0], ybnd[1]]
    cmap = cm.RdBu_r
    for k in range(len(u)):

        data = np.ma.masked_where(mask < 0.5, u[k])

        fig, ax = plt.subplots(1)
        ax.cla()
        ax.set_aspect("equal")
        img = ax.imshow(
            data.data, cmap=cmap, extent=extent, vmin=min_plt[k], vmax=max_plt[k]
        )
        add_colorbar(img)
        ax.contour(
            data.data,
            nlevels,
            colors="k",
            extent=extent,
            linewidths=0.5,
            linestyles="solid",
        )
        coll = PatchCollection(
            [patches.Circle((0, 0), 0.5, linewidth=1, color="w", ec=None, fc="w")],
            zorder=10,
            match_original=True,
        )
        ax.add_collection(coll)
        plt.xlabel(r"$x / D$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$y / D$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(
            os.path.join(fdir, "{0:s}{1:s}.png".format(names[k], sfx)),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(1)
        ax.cla()
        ax.set_aspect("equal")
        cmap.set_bad(color="black")
        img = ax.imshow(
            data, cmap=cmap, extent=extent, vmin=min_plt[k], vmax=max_plt[k]
        )
        add_colorbar(img)
        coll = PatchCollection(
            [patches.Circle((0, 0), 0.5, linewidth=1, color="w", ec=None, fc="w")],
            zorder=10,
            match_original=True,
        )
        ax.add_collection(coll)
        plt.xlabel(r"$x / D$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$y / D$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(
            os.path.join(fdir, "{0:s}{1:s}_masked.png".format(names[k], sfx)),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

    umag = np.sqrt(sum(map(lambda x: x * x, u)))
    data = np.ma.masked_where(mask < 0.5, umag)
    cmap = cm.viridis
    max_plt = 1.5
    fig, ax = plt.subplots(1)
    ax.cla()
    ax.set_aspect("equal")
    img = ax.imshow(data.data, cmap=cmap, extent=extent, vmin=0, vmax=max_plt)
    add_colorbar(img)
    plt.contour(
        data.data,
        nlevels,
        colors="k",
        extent=extent,
        linewidths=0.5,
        linestyles="solid",
    )
    coll = PatchCollection(
        [patches.Circle((0, 0), 0.5, linewidth=1, color="w", ec=None, fc="w")],
        zorder=10,
        match_original=True,
    )
    ax.add_collection(coll)
    plt.xlabel(r"$x / D$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$y / D$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(
        os.path.join(fdir, "{0:s}{1:s}.png".format(names[-1], sfx)),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(1)
    ax.cla()
    ax.set_aspect("equal")
    cmap = cm.viridis
    cmap.set_bad(color="black")
    img = ax.imshow(data, cmap="viridis", extent=extent, vmin=0, vmax=max_plt)
    add_colorbar(img)
    coll = PatchCollection(
        [patches.Circle((0, 0), 0.5, linewidth=1, color="w", ec=None, fc="w")],
        zorder=10,
        match_original=True,
    )
    ax.add_collection(coll)
    plt.xlabel(r"$x / D$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$y / D$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(
        os.path.join(fdir, "{0:s}{1:s}_masked.png".format(names[-1], sfx)),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Perform cylinder analysis")
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        help="Make velocity pseudo-color plots",
        action="store_true",
    )
    parser.add_argument(
        "-r" ",--results_directory",
        dest="results_directory",
        help="Directory containing the results to analyze",
        type=str,
        default="cylinder/results",
    )
    args = parser.parse_args()

    # Setup
    rdir = os.path.abspath(args.results_directory)
    fdirs = sorted(
        [
            os.path.join(rdir, d)
            for d in os.listdir(rdir)
            if os.path.isdir(os.path.join(rdir, d))
        ]
    )
    odir = os.path.abspath("./plots")
    fmt = "png"

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    lfdirs = fdirs[rank::nprocs]

    if (rank == 0) and (not os.path.exists(odir)):
        os.makedirs(odir)
    comm.Barrier()

    # Loop on all directories
    elst = []
    lname = "run.log"
    for cnt, fdir in enumerate(lfdirs):

        # Load the results file
        print("Analyzing {0:s}".format(fdir))
        casename = os.path.basename(fdir)
        fname = os.path.join(fdir, "data.npz")
        data = np.load(fname)
        result = data["result"]
        out = data["out"]
        original = data["original"]
        mask = data["mask"]
        mbox = data["mbox"]
        fraction = np.float(data["fraction"])

        # Load the interpolated image
        iname = os.path.join(fdir, "interpolated.npz")
        idata = np.load(iname)
        interp = idata["interpolated"]

        # Data
        U0 = [np.squeeze(original[c, :, :]) for c in range(2)]
        Ur = [np.squeeze(result[c, :, :]) for c in range(2)]
        Ui = [np.squeeze(interp[c, :, :]) for c in range(2)]

        # Rescale the data to the original input data
        lbnd, rbnd = get_scaling(os.path.join(fdir, lname))
        U0 = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(U0)]
        Ur = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(Ur)]
        Ui = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(Ui)]

        # Get the errors
        r_errors = [np.sqrt(np.mean((U0[k] - Ur[k]) ** 2)) for k in range(len(U0))]
        i_errors = [np.sqrt(np.mean((U0[k] - Ui[k]) ** 2)) for k in range(len(U0))]

        elst.append(
            pd.DataFrame(
                {
                    "fraction": fraction,
                    "mbox": mbox,
                    "ur_error": r_errors[0],
                    "vr_error": r_errors[1],
                    "ur_merror": r_errors[0] / np.sqrt(fraction / 100),
                    "vr_merror": r_errors[1] / np.sqrt(fraction / 100),
                    "ui_error": i_errors[0],
                    "vi_error": i_errors[1],
                    "ui_merror": i_errors[0] / np.sqrt(fraction / 100),
                    "vi_merror": i_errors[1] / np.sqrt(fraction / 100),
                    "casename": casename,
                },
                index=[cnt],
            )
        )

        # Plots of the velocities and velocity magnitude
        if args.plot:
            make_velocity_plots(fdir, U0, mask[0, :, :], "0")
            make_velocity_plots(fdir, Ur, mask[0, :, :], "r")
            make_velocity_plots(fdir, Ui, mask[0, :, :], "i")

    # Gather all to root processor
    elst = comm.gather(elst, root=0)
    comm.Barrier()

    # Save the errors and plot them
    if rank == 0:
        edf = pd.concat([item for sublist in elst for item in sublist])

        edf.sort_values(by=["fraction", "mbox", "casename"], inplace=True)
        edf.to_csv(os.path.join(rdir, "errors.dat"), index=False)

        mean_error = edf.groupby(["mbox"]).mean()
        std_error = edf.groupby(["mbox"]).std()

        plt.figure(0)
        ax = plt.gca()
        plt.errorbar(
            mean_error.index,
            mean_error.ur_merror,
            # yerr=std_error.ur_merror,
            color=cmap[0],
            mec=cmap[0],
            mfc=cmap[0],
            marker=markertype[0],
            ms=10,
            capsize=3,
        )
        plt.errorbar(
            mean_error.index,
            mean_error.ui_merror,
            # yerr=std_error.ui_merror,
            color=cmap[1],
            mec=cmap[1],
            mfc=cmap[1],
            marker=markertype[1],
            ms=10,
            capsize=3,
        )
        plt.xlabel(r"$L_m / D$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$L_2(u)$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.14)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "cyl_error_u.{0:s}".format(fmt)), format=fmt, dpi=300
        )

        plt.figure(1)
        ax = plt.gca()
        plt.errorbar(
            mean_error.index,
            mean_error.vr_merror,
            # yerr=std_error.vr_merror,
            color=cmap[0],
            mec=cmap[0],
            mfc=cmap[0],
            marker=markertype[0],
            ms=10,
            capsize=3,
        )
        plt.errorbar(
            mean_error.index,
            mean_error.vi_merror,
            # yerr=std_error.vi_merror,
            color=cmap[1],
            mec=cmap[1],
            mfc=cmap[1],
            marker=markertype[1],
            ms=10,
            capsize=3,
        )
        plt.xlabel(r"$L_m / D$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$L_2(v)$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.14)
        plt.gcf().subplots_adjust(left=0.16)
        plt.savefig(
            os.path.join(odir, "cyl_error_v.{0:s}".format(fmt)), format=fmt, dpi=300
        )

        # output timer
        end = time.time() - start
        print(
            "Elapsed time "
            + str(timedelta(seconds=end))
            + " (or {0:f} seconds)".format(end)
        )
