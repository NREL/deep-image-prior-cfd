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
plt.rc("text", usetex=True)


# ========================================================================
#
# Functions
#
# ========================================================================
def taylor_microscale(u, dx, axis=0):
    """Calculate the Taylor microscale

    :math:`\\lambda = \\sqrt{\\frac{\\langle u^2\\rangle}{\\left\\langle \\left( \\frac{\\partial u}{\\partial x}\\right)^2\\right\\rangle}}`

    :param u: velocity field
    :type u: array
    :param dx: grid spacing
    :type dx: double
    :param axis: direction of derivative
    :type axis: int
    :return: Taylor microscale
    :rtype: double
    """
    return np.sqrt(np.mean(u ** 2) / np.mean(np.gradient(u, dx, axis=axis) ** 2))


# ========================================================================
def calc_urms(u, v, w):
    """Calculate urms

    :math:`u' = \\sqrt{\\frac{\\left\\langle u^2 + v^2 + w^2 \\right\\rangle}{3}}`

    :param u: x-velocity field
    :type u: array
    :param v: y-velocity field
    :type v: array
    :param w: z-velocity field
    :type w: array
    :return: urms
    :rtype: double
    """
    return np.sqrt(np.mean(u ** 2 + v ** 2 + w ** 2) / 3)


# ========================================================================
def energy_spectra(u, v):
    """ Calculate 3D energy spectra

    The 3D energy spectrum is defined as (see Eq. 6.188 in Pope):

    - :math:`E_{3D}(k) = \\frac{1}{2} \\int_S \\phi_{ii}(k_0,k_1,k_2) \\mathrm{d}S(k)`

    where :math:`k=\\sqrt{k_0^2 + k_1^2 + k_2^2}` and
    :math:`\\phi_{ii}(k_0,k_1,k_2) = u_i u_i` (velocities
    in Fourier space) is filtered so that only valid wavenumber
    combinations are counted.

    .. note::

        For the 3D spectrum, the integral is approximated by averaging
        :math:`\\phi_{ii}(k_0,k_1,k_2)` over a binned :math:`k` and
        multiplying by the surface area of the sphere at :math:`k`. The
        bins are defined by rounding the wavenumber :math:`k` to the
        closest integer. An average of every :math:`k` in each bin is
        then used as the return value for that bin.

    :param u: x-velocity field
    :type u: array
    :param v: y-velocity field
    :type v: array
    :return: wavenumber bins
    :rtype: array
    :return: energy
    :rtype: array
    """

    # FFT velocities
    N = u.shape
    uf = np.fft.fftn(u)
    vf = np.fft.fftn(v)

    # Wavenumbers
    k = [np.fft.fftfreq(uf.shape[0]) * N[0], np.fft.fftfreq(vf.shape[1]) * N[1]]
    kmax = [k[0].max(), k[1].max()]
    K = np.meshgrid(k[0], k[1], indexing="ij")
    kmag = np.sqrt(K[0] ** 2 + K[1] ** 2)
    kbins = np.hstack((-1e-16, np.arange(0.5, N[0] // 2 - 1), N[0] // 2 - 1))

    # Energy in Fourier space
    Ef = 0.5 / (np.prod(N) ** 2) * (np.absolute(uf) ** 2 + np.absolute(vf) ** 2)

    # Filter the data with ellipsoid filter
    ellipse = (K[0] / kmax[0]) ** 2 + (K[1] / kmax[1]) ** 2 > 1.0
    Ef[ellipse] = np.nan
    K[0][ellipse] = np.nan
    K[1][ellipse] = np.nan
    kmag[ellipse] = np.nan

    # Multiply spectra by the surface area of the sphere at kmag.
    E3D = 4.0 * np.pi * kmag ** 2 * Ef

    # Binning
    whichbin = np.digitize(kmag.flat, kbins, right=True)
    ncount = np.bincount(whichbin)

    # Average in each wavenumber bin
    E = np.zeros(len(kbins) - 1)
    kavg = np.zeros(len(kbins) - 1)
    for k, n in enumerate(range(1, len(kbins))):
        whichbin_idx = whichbin == n
        E[k] = np.mean(E3D.flat[whichbin_idx])
        kavg[k] = np.mean(kmag.flat[whichbin_idx])
    E[E < 1e-13] = 0.0

    return kavg, E


# ========================================================================
def normalized_velocity_derivative(u, dx, axis=0):
    """Calculate normalized velocity derivative

    :math:`Z = \\left. \\frac{\\partial u}{\\partial x} \\middle/ \\left\\langle \\left( \\frac{\\partial u}{\\partial x} \\right)^2 \\right\\rangle^{1/2} \\right.`

    :param u: velocity field
    :type u: array
    :param dx: grid spacing
    :type dx: double
    :param axis: direction of derivative
    :type axis: int
    :return: Z
    :rtype: array
    """
    dudx = np.gradient(u, dx, axis=axis)
    return dudx / np.sqrt(np.mean(dudx ** 2))


# ========================================================================
def structure_function_exponent(u, order):
    """Calculate the exponent of the structure function of a given order

    Find the exponent of
    :math:`S_n (r) \sim r^{\zeta_n}`

    where :math:`S_n (r) = \\left\\langle (\\Delta_r u)^n \\right\\rangle`
    and :math:`\\Delta_r u = u(\\mathbf{x}+\\mathbf{e_1} r,t) - u(\\mathbf{x},t)`

    :param u: velocity field
    :type u: array
    :param order: structure function order
    :type order: int
    :return: exponent
    :rtype: double
    """
    rs = np.arange(1, 12)
    Sn = np.zeros(len(rs))
    for r in rs:
        Sn[r - 1] = np.mean(np.fabs(u[r:, :] - u[:-r, :]) ** order)

    # Sn ~ b * rs ** m
    p = np.polyfit(np.log(rs[1:]), np.log(Sn[1:]), 1)
    m = p[0]
    b = np.exp(p[1])

    return m


# ========================================================================
def get_scaling(logname):
    with open(logname, "r") as f:
        for line in f:
            if "Image name" in line:
                imgname = line.split()[-1]
                imgdir = os.path.dirname(imgname)
                base = os.path.splitext(os.path.basename(imgname))[0]

    imgdir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "turbulence", "images_64"
    )
    sname = os.path.join(imgdir, "scalings.dat")
    df = pd.read_csv(sname)

    sdf = df[df.filename == base]
    return (
        [np.float(sdf.u_lbnd), np.float(sdf.v_lbnd), np.float(sdf.w_lbnd)],
        [np.float(sdf.u_rbnd), np.float(sdf.v_rbnd), np.float(sdf.w_rbnd)],
    )


# ========================================================================
def make_velocity_plots(fdir, u, mask, sfx):
    """Make some plots of velocity and velocity magnitude"""

    urms_theory = np.sqrt(2)
    names = ["u", "v", "w", "umag"]
    max_plt = 3
    for k in range(len(u)):

        data = np.ma.masked_where(mask < 0.5, u[k] / urms_theory)

        plt.figure(0)
        plt.clf()
        ax = plt.gca()
        plt.imshow(
            data.data, cmap="RdBu_r", extent=[0, 1, 0, 1], vmax=max_plt, vmin=-max_plt
        )
        plt.colorbar()
        plt.xlabel(r"$x / L$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$y / L$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(
            os.path.join(fdir, "{0:s}{1:s}.png".format(names[k], sfx)),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.figure(1)
        plt.clf()
        ax = plt.gca()
        cmap = cm.RdBu_r
        cmap.set_bad(color="black")
        plt.imshow(data, cmap=cmap, extent=[0, 1, 0, 1], vmax=max_plt, vmin=-max_plt)
        plt.colorbar()
        plt.xlabel(r"$x / L$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$y / L$", fontsize=22, fontweight="bold")
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
    data = np.ma.masked_where(mask < 0.5, umag / urms_theory)
    max_plt = 3
    plt.figure(0)
    plt.clf()
    ax = plt.gca()
    plt.imshow(data.data, cmap="viridis", extent=[0, 1, 0, 1], vmax=max_plt, vmin=0)
    plt.colorbar()
    plt.xlabel(r"$x / L$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$y / L$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(
        os.path.join(fdir, "{0:s}{1:s}.png".format(names[-1], sfx)),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.figure(1)
    plt.clf()
    ax = plt.gca()
    cmap = cm.viridis
    cmap.set_bad(color="black")
    plt.imshow(data, cmap="viridis", extent=[0, 1, 0, 1], vmax=max_plt, vmin=0)
    plt.colorbar()
    plt.xlabel(r"$x / L$", fontsize=22, fontweight="bold")
    plt.ylabel(r"$y / L$", fontsize=22, fontweight="bold")
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="Perform turbulence analysis")
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
        default="turbulence/results",
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
    eklst = []
    sflst = []
    mean_Zs = [0, 0, 0]
    lname = "run.log"
    for cnt, fdir in enumerate(lfdirs):

        # Load the results file
        print("Analyzing {0:s}".format(fdir))
        casename = os.path.basename(fdir)
        fname = os.path.join(fdir, "data.npz")
        data = np.load(fname)
        result = data["result"]
        original = data["original"]
        mask = data["mask"]
        mbox = data["mbox"]
        fraction = np.float(data["fraction"])

        # Load the interpolated image
        iname = os.path.join(fdir, "interpolated.npz")
        idata = np.load(iname)
        interp = idata["interpolated"]

        # Data
        L = np.array([2 * np.pi, 2 * np.pi])
        N = np.array([result.shape[1], result.shape[2]])
        dx = L / N
        x = np.arange(dx[0] / 2, L[0], dx[0])
        y = np.arange(dx[0] / 2, L[1], dx[1])
        X = np.meshgrid(x, y)
        U0 = [np.squeeze(original[c, :, :]) for c in range(3)]
        Ur = [np.squeeze(result[c, :, :]) for c in range(3)]
        Ui = [np.squeeze(interp[c, :, :]) for c in range(3)]

        # Rescale the data to the original input data
        lbnd, rbnd = get_scaling(os.path.join(fdir, lname))
        U0 = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(U0)]
        Ur = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(Ur)]
        Ui = [u * (rbnd[k] - lbnd[k]) + lbnd[k] for k, u in enumerate(Ui)]

        # urms
        urms0 = calc_urms(U0[0], U0[1], 0)
        urmsr = calc_urms(Ur[0], Ur[1], 0)
        urmsi = calc_urms(Ui[0], Ui[1], 0)

        # Taylor microscale
        lambda0 = taylor_microscale(U0[0], dx[0])
        lambdar = taylor_microscale(Ur[0], dx[0])
        lambdai = taylor_microscale(Ui[0], dx[0])

        # Spectra
        kbins, Ek0 = energy_spectra(U0[0], U0[1])
        _, Ekr = energy_spectra(Ur[0], Ur[1])
        _, Eki = energy_spectra(Ui[0], Ui[1])

        # Velocity derivatives
        Z0 = normalized_velocity_derivative(U0[0], dx[0])
        Zr = normalized_velocity_derivative(Ur[0], dx[0])
        Zi = normalized_velocity_derivative(Ui[0], dx[0])
        mean_Zs = [mean_Zs[k] + Z for k, Z in enumerate([Z0, Zr, Zi])]

        # Skewness
        S0 = np.mean(Z0 ** 3)
        Sr = np.mean(Zr ** 3)
        Si = np.mean(Zi ** 3)

        # Structure function exponents
        sf_orders = np.arange(2, 11)
        zetas0 = np.zeros(sf_orders.shape)
        zetasr = np.zeros(sf_orders.shape)
        zetasi = np.zeros(sf_orders.shape)
        for k, order in enumerate(sf_orders):
            zetas0[k] = structure_function_exponent(U0[0], order)
            zetasr[k] = structure_function_exponent(Ur[0], order)
            zetasi[k] = structure_function_exponent(Ui[0], order)

        # Get the errors
        r_errors = [np.sqrt(np.mean((U0[k] - Ur[k]) ** 2)) for k in range(len(U0))]
        i_errors = [np.sqrt(np.mean((U0[k] - Ui[k]) ** 2)) for k in range(len(U0))]

        # Save all the statistics
        np.savez_compressed(
            os.path.join(fdir, "statistics.npz"),
            N=N,
            dx=dx,
            L=L,
            urms0=urms0,
            urmsr=urmsr,
            urmsi=urmsi,
            lambda0=lambda0,
            lambdar=lambdar,
            lambdai=lambdai,
            lmask=mbox * dx[0],
            kbins=kbins,
            Ek0=Ek0,
            Ekr=Ekr,
            Eki=Eki,
            Z0=Z0,
            Zr=Zr,
            Zi=Zi,
            S0=S0,
            Sr=Sr,
            Si=Si,
            sf_orders=sf_orders,
            zetas0=zetas0,
            zetasr=zetasr,
            zetasi=zetasi,
            r_errors=r_errors,
            i_errors=i_errors,
            fraction=fraction,
            casename=casename,
        )

        slst.append(
            {
                "N": N[0],
                "fraction": fraction,
                "lmask": mbox * dx[0],
                "urms0": urms0,
                "urmsr": urmsr,
                "urmsi": urmsi,
                "lambda0": lambda0,
                "lambdar": lambdar,
                "lambdai": lambdai,
                "S0": S0,
                "Sr": Sr,
                "Si": Si,
                "ur_error": r_errors[0],
                "vr_error": r_errors[1],
                "wr_error": r_errors[2],
                "ur_merror": r_errors[0] / np.sqrt(fraction / 100),
                "vr_merror": r_errors[1] / np.sqrt(fraction / 100),
                "wr_merror": r_errors[2] / np.sqrt(fraction / 100),
                "ui_error": i_errors[0],
                "vi_error": i_errors[1],
                "wi_error": i_errors[2],
                "ui_merror": i_errors[0] / np.sqrt(fraction / 100),
                "vi_merror": i_errors[1] / np.sqrt(fraction / 100),
                "wi_merror": i_errors[2] / np.sqrt(fraction / 100),
                "casename": casename,
            }
        )

        eklst.append(
            pd.DataFrame(
                {
                    "N": N[0],
                    "fraction": fraction,
                    "lmask": mbox * dx[0],
                    "kbins": kbins,
                    "Ek0": Ek0,
                    "Ekr": Ekr,
                    "Eki": Eki,
                    "casename": casename,
                }
            )
        )
        sflst.append(
            pd.DataFrame(
                {
                    "N": N[0],
                    "fraction": fraction,
                    "lmask": mbox * dx[0],
                    "sf_orders": sf_orders,
                    "zetas0": zetas0,
                    "zetasr": zetasr,
                    "zetasi": zetasi,
                    "casename": casename,
                }
            )
        )

        # Save binary files for Pele
        u0 = U0[0].flatten()
        v0 = U0[1].flatten()
        df = pd.DataFrame(
            {
                "x": X[0].flatten(),
                "y": X[1].flatten(),
                "z": np.zeros(X[1].flatten().shape),
                "u": u0,
                "v": v0,
                "w": np.zeros(u0.shape),
            }
        )
        df.sort_values(by=["z", "y", "x"], inplace=True)
        df.values.tofile(os.path.join(fdir, "original.in"))

        ur = Ur[0].flatten()
        vr = Ur[1].flatten()
        df = pd.DataFrame(
            {
                "x": X[0].flatten(),
                "y": X[1].flatten(),
                "z": np.zeros(X[1].flatten().shape),
                "u": ur,
                "v": vr,
                "w": np.zeros(ur.shape),
            }
        )
        df.sort_values(by=["z", "y", "x"], inplace=True)
        df.values.tofile(os.path.join(fdir, "result.in"))

        ui = Ui[0].flatten()
        vi = Ui[1].flatten()
        df = pd.DataFrame(
            {
                "x": X[0].flatten(),
                "y": X[1].flatten(),
                "z": np.zeros(X[1].flatten().shape),
                "u": ui,
                "v": vi,
                "w": np.zeros(ur.shape),
            }
        )
        df.sort_values(by=["z", "y", "x"], inplace=True)
        df.values.tofile(os.path.join(fdir, "interp.in"))

        # Plots of the velocities and velocity magnitude
        if args.plot:
            make_velocity_plots(fdir, U0, mask[0, :, :], "0")
            make_velocity_plots(fdir, Ur, mask[0, :, :], "r")
            make_velocity_plots(fdir, Ui, mask[0, :, :], "i")

    # Gather all to root processor
    slst = comm.gather(slst, root=0)
    eklst = comm.gather(eklst, root=0)
    sflst = comm.gather(sflst, root=0)
    mean_Zs = comm.gather(mean_Zs, root=0)
    comm.Barrier()

    # Store stuff
    if rank == 0:
        stats = pd.DataFrame([item for sublist in slst for item in sublist])
        stats.sort_values(by=["N", "fraction", "lmask", "casename"], inplace=True)
        stats.to_csv(os.path.join(rdir, "statistics.dat"), index=False)
        ekdf = pd.concat([item for sublist in eklst for item in sublist])
        ekdf.sort_values(by=["N", "fraction", "lmask", "casename"], inplace=True)
        ekdf.to_csv(os.path.join(rdir, "spectra.dat"), index=False)
        sfdf = pd.concat([item for sublist in sflst for item in sublist])
        sfdf.sort_values(by=["N", "fraction", "lmask", "casename"], inplace=True)
        sfdf.to_csv(os.path.join(rdir, "structure.dat"), index=False)
        mean_Zs = [sum(mz) / len(fdirs) for mz in zip(*mean_Zs)]
        np.savez_compressed(
            os.path.join(rdir, "mean_Zs.npz"),
            Z0=mean_Zs[0],
            Zr=mean_Zs[1],
            Zi=mean_Zs[2],
        )

        # output timer
        end = time.time() - start
        print(
            "Elapsed time "
            + str(timedelta(seconds=end))
            + " (or {0:f} seconds)".format(end)
        )
