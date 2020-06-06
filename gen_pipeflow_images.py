#!/usr/bin/env python3
"""Generate images from raw pipeflow data
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
import pandas as pd
import struct

import matplotlib.pyplot as plt

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
        description="Generate images from raw pipeflow data"
    )
    args = parser.parse_args()

    # Setup
    tpath = os.path.abspath("pipeflow")
    rdir = os.path.join(tpath, "raw")
    odir = os.path.join(tpath, "images")
    sname = os.path.abspath(os.path.join(odir, "scalings.dat"))
    if os.path.exists(odir):
        for f in glob.glob(os.path.join(odir, "pf*.png")):
            os.remove(f)
    else:
        os.makedirs(odir)

    # Load the raw data
    with open(os.path.join(rdir, "InflowPC.bin"), mode="rb") as f:
        fc = f.read()
        cnt = 0

        bytes_int = 4
        bytes_double = 8

        # Read ints
        n = 3
        nb = 3 * bytes_int
        nt, nr, nth = struct.unpack("i" * n, fc[cnt : cnt + nb])
        cnt += nb

        # Read doubles
        n = 2
        nb = n * bytes_double
        freq, times = struct.unpack("d" * n, fc[cnt : cnt + nb])
        cnt += nb

        # time grid
        times = np.arange(nt) * freq

        # radial grid
        n = nr + 1
        nb = n * bytes_double
        r = np.array(struct.unpack("d" * n, fc[cnt : cnt + nb]))
        cnt += nb
        rh = 0.5 * (r[1:] + r[:-1])

        # azimuthal grid
        n = nth + 1
        nb = n * bytes_double
        theta = np.array(struct.unpack("d" * n, fc[cnt : cnt + nb]))
        cnt += nb
        thetah = 0.5 * (theta[1:] + theta[:-1])

        # velocities
        n = nt * nr * nth
        nb = n * bytes_double
        uz = np.array(struct.unpack("d" * n, fc[cnt : cnt + nb])).reshape(nt, nr, nth)
        cnt += nb

        n = nt * nr * nth
        nb = n * bytes_double
        ur = np.array(struct.unpack("d" * n, fc[cnt : cnt + nb])).reshape(nt, nr, nth)
        cnt += nb

        n = nt * nr * nth
        nb = n * bytes_double
        ut = np.array(struct.unpack("d" * n, fc[cnt : cnt + nb])).reshape(nt, nr, nth)
        cnt += nb

    # Drop the last radial value to make the image even sized (and those are zero anyway)
    ut = ut[:, :-1, :]
    uz = uz[:, :-1, :]
    ur = ur[:, :-1, :]

    # Save slices of the data
    img_np = np.zeros((nr - 1, nth, 3))
    lst = []
    for cnt in range(nt):

        img_np[:, :, 0] = np.take(uz, cnt, axis=0)
        img_np[:, :, 1] = np.take(ur, cnt, axis=0)
        img_np[:, :, 2] = np.take(ut, cnt, axis=0)

        # Scale to between zero and one
        lbnd = np.min(img_np, axis=(0, 1))
        rbnd = np.max(img_np, axis=(0, 1))
        for k in range(3):
            img_np[:, :, k] = (img_np[:, :, k] - lbnd[k]) / (rbnd[k] - lbnd[k])

        # Save image and numpy array
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        sfx = "pf{0:04d}".format(cnt)
        np.savez_compressed(os.path.join(odir, sfx) + ".npz", img=img_np)
        img_pil.save(os.path.join(odir, sfx) + ".png")

        # Log the scaling
        lst.append(
            {
                "filename": sfx,
                "u_lbnd": lbnd[0],
                "u_rbnd": rbnd[0],
                "v_lbnd": lbnd[1],
                "v_rbnd": rbnd[1],
                "w_lbnd": lbnd[2],
                "w_rbnd": rbnd[2],
            }
        )

    scalings = pd.DataFrame(lst)
    scalings.to_csv(sname, index=False)

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )

    # # radial plot
    # theta, rad = np.meshgrid(thetah, rh)
    # theta = np.hstack((theta, theta[:, 0][:, np.newaxis]))
    # rad = np.hstack((rad, rad[:, 0][:, np.newaxis]))
    # u = np.hstack((ut[0, :, :], ut[0, :, 0][:, np.newaxis]))
    # print(ut[0, -1, :])
    # print(ur[0, -1, :])
    # print(uz[0, -1, :])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, polar="True")
    # ax.pcolormesh(theta, rad, u, shading="gouraud")
    # ax.spines["polar"].set_visible(False)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # plt.tight_layout()
    # plt.savefig("c.png")
