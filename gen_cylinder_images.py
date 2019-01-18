#!/usr/bin/env python3
"""Generate images from raw cylinder flow data
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import glob
import argparse
import numpy as np
import time
from datetime import timedelta
from PIL import Image
import netCDF4 as nc
import pandas as pd
from scipy.interpolate import griddata


# ========================================================================
#
# Functions
#
# ========================================================================
def get_var_front(fname):
    """Get variables on the front from Exodus file."""

    # Load the data
    dat = nc.Dataset(fname)

    # Load the sideset and variable names
    ssn = ["%s" % nc.chartostring(ss) for ss in dat.variables["ss_names"][:]]
    vn = ["%s" % nc.chartostring(nn) for nn in dat.variables["name_nod_var"][:]]

    # Get sideset and variable indices
    idx_11 = ssn.index("side11")
    idx_21 = ssn.index("side21")
    idx_ux = vn.index("velocity_x")
    idx_uz = vn.index("velocity_z")
    idx_ib = vn.index("iblank")

    # Get element index
    elem_idx_11 = dat.variables["elem_ss{0:d}".format(idx_11 + 1)][:] - 1
    elem_idx_21 = dat.variables["elem_ss{0:d}".format(idx_21 + 1)][:] - 1

    # Get node connectivity
    connect1_11 = dat.variables["connect1"][elem_idx_11].flatten() - 1
    connect2_21 = (
        dat.variables["connect2"][elem_idx_21 - np.min(elem_idx_21)].flatten() - 1
    )

    # Get coordinates
    coord_11 = (
        dat.variables["coordx"][connect1_11],
        dat.variables["coordy"][connect1_11],
        dat.variables["coordz"][connect1_11],
    )
    coord_21 = (
        dat.variables["coordx"][connect2_21],
        dat.variables["coordy"][connect2_21],
        dat.variables["coordz"][connect2_21],
    )

    # Subset on the nodes on the face
    actual_idx_11 = np.where(coord_11[1] <= 0.0 + 1e-10)
    coord_11 = (
        coord_11[0][actual_idx_11],
        coord_11[1][actual_idx_11],
        coord_11[2][actual_idx_11],
    )
    connect1_11 = connect1_11[actual_idx_11]
    actual_idx_21 = np.where(coord_21[1] <= 0.0 + 1e-10)
    coord_21 = (
        coord_21[0][actual_idx_21],
        coord_21[1][actual_idx_21],
        coord_21[2][actual_idx_21],
    )
    connect2_21 = connect2_21[actual_idx_21]

    # Get variables at those nodes
    ux_11 = dat.variables["vals_nod_var{0:d}".format(idx_ux + 1)][-1, connect1_11]
    uz_11 = dat.variables["vals_nod_var{0:d}".format(idx_uz + 1)][-1, connect1_11]
    ib_11 = dat.variables["vals_nod_var{0:d}".format(idx_ib + 1)][-1, connect1_11]
    ux_21 = dat.variables["vals_nod_var{0:d}".format(idx_ux + 1)][-1, connect2_21]
    uz_21 = dat.variables["vals_nod_var{0:d}".format(idx_uz + 1)][-1, connect2_21]
    ib_21 = dat.variables["vals_nod_var{0:d}".format(idx_ib + 1)][-1, connect2_21]

    # Concatenate all of this
    colnames = ["x", "z", "ux", "uz", "iblank"]
    df_11 = pd.DataFrame(
        data=np.vstack((coord_11[0], coord_11[2], ux_11, uz_11, ib_11)).T,
        columns=colnames,
    )
    df_21 = pd.DataFrame(
        data=np.vstack((coord_21[0], coord_21[2], ux_21, uz_21, ib_21)).T,
        columns=colnames,
    )
    df = pd.concat([df_11, df_21], ignore_index=True)

    # Only keep values where iblank = 1
    df = df.loc[df["iblank"] == 1]

    df.x[np.fabs(df.x) < 1e-14] = 0.0  # make true zeros
    df.z[np.fabs(df.z) < 1e-14] = 0.0  # make true zeros
    df = df.sort_values(by=["x", "z"])
    df = df.drop_duplicates(subset=["x", "z"])

    return df.reset_index()


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate images (and masks) for cylinder flow data"
    )
    parser.add_argument(
        "-r", "--resolution", dest="res", help="Image resolution", type=int, default=128
    )
    args = parser.parse_args()

    # Setup
    seed = 89457364
    np.random.seed(seed)
    cpath = os.path.abspath("cylinder")
    rdir = os.path.join(cpath, "raw")
    odir = os.path.join(cpath, "images_{0:d}".format(args.res))
    lname = os.path.abspath(os.path.join(odir, "log.dat"))
    sname = os.path.abspath(os.path.join(odir, "scalings.dat"))
    if os.path.exists(odir):
        for f in glob.glob(os.path.join(odir, "*.png")):
            os.remove(f)
    else:
        os.makedirs(odir)

    # Load the raw data
    ename = os.path.join(rdir, "cyl_hypre.e-s.001")
    dname = os.path.join(rdir, "cyl_hypre.dat")
    if os.path.isfile(dname):
        df = pd.read_csv(dname)
    else:
        df = get_var_front(ename)
        df.to_csv(os.path.join(rdir, "cyl_hypre.dat"), index=False)

    # Interpolate on to a regular grid
    xbnd = np.array([-2, 15])
    x, dx = np.linspace(xbnd[0], xbnd[1], args.res, retstep=True)
    resy = args.res // 2
    ybnd = [-0.5 * resy * dx, 0.5 * resy * dx]
    L = np.array([xbnd[1] - xbnd[0], ybnd[1] - ybnd[0]])
    y = np.linspace(ybnd[0], ybnd[1], resy)
    X, Y = np.meshgrid(x, y)
    ux = griddata((df.x, df.z), df.ux, (X, Y), method="linear")
    uz = griddata((df.x, df.z), df.uz, (X, Y), method="linear")

    # Velocity in cylinder is zero
    radius = 0.5
    cyl_center = np.array([0.0, 0.0])
    cyl_res = int(2 * radius / dx)
    cyl = np.sqrt(X ** 2 + Y ** 2) <= radius
    ux[cyl] = 0
    uz[cyl] = 0

    # Bounds for the scaling
    ux_lbnd = np.min(ux)
    ux_rbnd = np.max(ux)
    uz_lbnd = np.min(uz)
    uz_rbnd = np.max(uz)

    # Make the image and scale to between zero and one
    img_np = np.zeros((resy, args.res, 3))
    img_np[:, :, 0] = (ux - ux_lbnd) / (ux_rbnd - ux_lbnd)
    img_np[:, :, 1] = (uz - uz_lbnd) / (uz_rbnd - uz_lbnd)

    # Save image and numpy array
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    sfx = "cyl"
    np.savez_compressed(os.path.join(odir, sfx) + ".npz", img=img_np, x=x, y=y)
    img_pil.save(os.path.join(odir, sfx) + ".png")

    # Log the scaling
    scalings = pd.DataFrame(
        [
            {
                "filename": sfx,
                "ux_lbnd": ux_lbnd,
                "ux_rbnd": ux_rbnd,
                "uz_lbnd": uz_lbnd,
                "uz_rbnd": uz_rbnd,
            }
        ]
    )
    scalings.to_csv(sname, index=False)

    # Generate masks
    nsamples = 40
    mask_cyl_ratios = [0.5, 1, 2, 3, 4, 5]
    valid_x = np.array([int((cyl_center[0] - xbnd[0] + 3) / dx), int((L[0] - 2) / dx)])
    valid_y = resy // 2 + np.array([-int(1 / dx), int(1 / dx)])

    cnt = 0
    lst = []
    for mask_cyl_ratio in mask_cyl_ratios:
        for nsample in range(nsamples):
            xidx = np.random.randint(low=valid_x[0], high=valid_x[1])
            yidx = np.random.randint(low=valid_y[0], high=valid_y[1])

            mask_np = np.ones(img_np.shape)
            mask_size = int(mask_cyl_ratio * cyl_res)
            mask_np[
                yidx - mask_size // 2 : yidx + mask_size // 2,
                xidx - mask_size // 2 : xidx + mask_size // 2,
                :,
            ] = 0

            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            fname = "mask{0:04d}.png".format(cnt)
            mask_pil.save(os.path.join(odir, fname))
            fraction = (1 - np.mean(mask_np[:, :, 0])) * 100
            lst.append(
                {"mbox": mask_cyl_ratio, "fraction": fraction, "filename": fname}
            )
            cnt += 1
    log = pd.DataFrame(lst)
    log.to_csv(lname, index=False)

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
