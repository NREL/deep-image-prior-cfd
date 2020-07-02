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
from mpi4py import MPI

# ========================================================================
#
# Some defaults variables
#
# ========================================================================
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
plt.rc("text", usetex=True)


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

    imgdir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "pipeflow", "images"
    )
    sname = os.path.join(imgdir, "scalings.dat")
    df = pd.read_csv(sname)

    sdf = df[df.filename == base]
    return (
        [np.float(sdf.u_lbnd), np.float(sdf.v_lbnd), np.float(sdf.w_lbnd)],
        [np.float(sdf.u_rbnd), np.float(sdf.v_rbnd), np.float(sdf.w_rbnd)],
    )


# ========================================================================
def make_velocity_plots(fdir, u, mask, theta, rad, sfx):
    """Make some plots of velocity and velocity magnitude"""

    urms_theory = 1.0
    names = ["uz", "ur", "ut", "umag"]
    max_plt = 100

    # close the periodic gap in theta
    l_theta = np.hstack((theta, theta[:, 0][:, np.newaxis]))
    l_rad = np.hstack((rad, rad[:, 0][:, np.newaxis]))

    for k in range(len(u)):

        l_u = np.hstack((u[k], u[k][:, 0][:, np.newaxis]))
        l_mask = np.hstack((mask, mask[:, 0][:, np.newaxis]))
        data = np.ma.masked_where(l_mask < 0.5, l_u / urms_theory)

        fig = plt.figure(0)
        plt.clf()
        ax = fig.add_subplot(111, polar="True")
        im = ax.pcolormesh(
            l_theta, l_rad, data.data, shading="gouraud", vmax=max_plt, vmin=-max_plt
        )
        ax.spines["polar"].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar(im)
        # plt.xlabel(r"$x / L$", fontsize=22, fontweight="bold")
        # plt.ylabel(r"$y / L$", fontsize=22, fontweight="bold")
        # plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        # plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(
            os.path.join(fdir, "{0:s}{1:s}.png".format(names[k], sfx)),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

        fig = plt.figure(1)
        plt.clf()
        cmap = cm.RdBu_r
        cmap.set_bad(color="black")
        ax = fig.add_subplot(111, polar="True")
        im = ax.pcolormesh(
            l_theta,
            l_rad,
            data,
            shading="gouraud",
            cmap=cmap,
            vmax=max_plt,
            vmin=-max_plt,
        )
        ax.spines["polar"].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar(im)
        # plt.xlabel(r"$x / L$", fontsize=22, fontweight="bold")
        # plt.ylabel(r"$y / L$", fontsize=22, fontweight="bold")
        # plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        # plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(
            os.path.join(fdir, "{0:s}{1:s}_masked.png".format(names[k], sfx)),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

    umag = np.sqrt(sum(map(lambda x: x * x, u)))
    umag = np.hstack((umag, umag[:, 0][:, np.newaxis]))
    data = np.ma.masked_where(l_mask < 0.5, umag / urms_theory)
    max_plt = 100
    fig = plt.figure(0)
    plt.clf()
    ax = fig.add_subplot(111, polar="True")
    im = ax.pcolormesh(
        l_theta, l_rad, data.data, shading="gouraud", vmax=max_plt, vmin=0
    )
    ax.spines["polar"].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(im)
    # plt.xlabel(r"$x / L$", fontsize=22, fontweight="bold")
    # plt.ylabel(r"$y / L$", fontsize=22, fontweight="bold")
    # plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    # plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(
        os.path.join(fdir, "{0:s}{1:s}.png".format(names[-1], sfx)),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    fig = plt.figure(1)
    plt.clf()
    cmap = cm.viridis
    cmap.set_bad(color="black")
    ax = fig.add_subplot(111, polar="True")
    im = ax.pcolormesh(
        l_theta, l_rad, data, shading="gouraud", cmap=cmap, vmax=max_plt, vmin=0
    )
    ax.spines["polar"].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(im)
    # plt.xlabel(r"$x / L$", fontsize=22, fontweight="bold")
    # plt.ylabel(r"$y / L$", fontsize=22, fontweight="bold")
    # plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    # plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(
        os.path.join(fdir, "{0:s}{1:s}_masked.png".format(names[-1], sfx)),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Perform pipeflow analysis")
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
        default="pipeflow/results",
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

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    lfdirs = fdirs[rank::nprocs]

    # Loop on all directories
    slst = []
    lname = "run.log"
    for cnt, fdir in enumerate(lfdirs):

        # Load the results file
        print("Analyzing {0:s}".format(fdir))
        casename = os.path.basename(fdir)
        fname = os.path.join(fdir, "data.npz")
        data = np.load(fname)
        _, _, nth = data["result"].shape
        zero = np.zeros((3, 1, nth))
        result = np.concatenate((data["result"], zero), axis=1)
        original = np.concatenate((data["original"], zero), axis=1)
        mask = np.concatenate((data["mask"], zero), axis=1)
        mbox = data["mbox"]
        _, nr, nth = result.shape
        rmax = 0.0008006168987282874

        fraction = np.float(data["fraction"])

        # Load the interpolated image
        iname = os.path.join(fdir, "interpolated.npz")
        idata = np.load(iname)
        interp = np.concatenate((idata["interpolated"], zero), axis=1)

        # Data
        theta = np.linspace(0, 2 * np.pi, nth + 1)
        thetah = 0.5 * (theta[1:] + theta[:-1])
        r = np.linspace(0, rmax, nr + 1)
        rh = 0.5 * (r[1:] + r[:-1])
        theta, rad = np.meshgrid(thetah, rh)
        U0 = [np.squeeze(original[c, :, :]) for c in range(3)]
        Ur = [np.squeeze(result[c, :, :]) for c in range(3)]
        Ui = [np.squeeze(interp[c, :, :]) for c in range(3)]

        # Rescale the data to the original input data
        lbnd, rbnd = get_scaling(os.path.join(fdir, lname))
        U0 = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(U0)]
        Ur = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(Ur)]
        Ui = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(Ui)]

        # Means
        uz_0 = np.mean(U0[0], axis=1)
        uz_r = np.mean(Ur[0], axis=1)
        uz_i = np.mean(Ui[0], axis=1)
        if args.plot:
            plt.figure(0)
            plt.clf()
            ax = plt.gca()
            for k, fld in enumerate([uz_0, uz_r, uz_i]):
                p = plt.plot(rh, uz_0, lw=2, color=cmap[k])
                p[0].set_dashes(dashseq[k])
            plt.xlabel(r"$r$", fontsize=22, fontweight="bold")
            plt.ylabel(r"$u$", fontsize=22, fontweight="bold")
            plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
            plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
            plt.gcf().subplots_adjust(bottom=0.14)
            plt.gcf().subplots_adjust(left=0.16)
            plt.savefig(
                os.path.join(fdir, "mean_profiles.png"), format="png", dpi=300,
            )

        # Get the errors
        r_errors = [np.sqrt(np.mean((U0[k] - Ur[k]) ** 2)) for k in range(len(U0))]
        i_errors = [np.sqrt(np.mean((U0[k] - Ui[k]) ** 2)) for k in range(len(U0))]

        # Save statistics
        np.savez_compressed(
            os.path.join(fdir, "statistics.npz"),
            nth=nth,
            nr=nr,
            thetah=thetah,
            rh=rh,
            uz_0=uz_0,
            uz_r=uz_r,
            uz_i=uz_i,
            r_errors=r_errors,
            i_errors=i_errors,
            fraction=fraction,
            casename=casename,
        )

        slst.append(
            {
                "nth": nth,
                "nr": nr,
                "thetah": thetah,
                "rh": rh,
                "fraction": fraction,
                "uzr_error": r_errors[0],
                "urr_error": r_errors[1],
                "utr_error": r_errors[2],
                "uzr_merror": r_errors[0] / np.sqrt(fraction / 100),
                "urr_merror": r_errors[1] / np.sqrt(fraction / 100),
                "utr_merror": r_errors[2] / np.sqrt(fraction / 100),
                "uzi_error": i_errors[0],
                "uri_error": i_errors[1],
                "uti_error": i_errors[2],
                "uzi_merror": i_errors[0] / np.sqrt(fraction / 100),
                "uri_merror": i_errors[1] / np.sqrt(fraction / 100),
                "uti_merror": i_errors[2] / np.sqrt(fraction / 100),
                "casename": casename,
            }
        )

        # Plots of the velocities and velocity magnitude
        if args.plot:
            make_velocity_plots(fdir, U0, mask[0, :, :], theta, rad, "0")
            make_velocity_plots(fdir, Ur, mask[0, :, :], theta, rad, "r")
            make_velocity_plots(fdir, Ui, mask[0, :, :], theta, rad, "i")

    # Gather all to root processor
    slst = comm.gather(slst, root=0)
    comm.Barrier()

    # Store stuff
    if rank == 0:
        stats = pd.DataFrame([item for sublist in slst for item in sublist])
        stats.sort_values(by=["fraction", "casename"], inplace=True)
        stats.to_csv(os.path.join(rdir, "statistics.dat"), index=False)

        # output timer
        end = time.time() - start
        print(
            "Elapsed time "
            + str(timedelta(seconds=end))
            + " (or {0:f} seconds)".format(end)
        )
