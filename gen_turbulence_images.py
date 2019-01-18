#!/usr/bin/env python3
"""Generate images from raw HIT data
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


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate images from raw HIT data")
    parser.add_argument(
        "-r", "--resolution", dest="res", help="Image resolution", type=int, default=64
    )
    args = parser.parse_args()

    # Setup
    tpath = os.path.abspath("turbulence")
    rdir = os.path.join(tpath, "raw")
    odir = os.path.join(tpath, "images_{0:d}".format(args.res))
    sname = os.path.abspath(os.path.join(odir, "scalings.dat"))
    if os.path.exists(odir):
        for f in glob.glob(os.path.join(odir, "hit*.png")):
            os.remove(f)
    else:
        os.makedirs(odir)

    # Load the raw data
    dat = np.fromfile(os.path.join(rdir, "hit_{0:d}.in".format(args.res)))
    u = dat[3::6].reshape((args.res, args.res, args.res), order="F")
    v = dat[4::6].reshape((args.res, args.res, args.res), order="F")
    w = dat[5::6].reshape((args.res, args.res, args.res), order="F")

    # Save slices of the data
    cnt = 0
    img_np = np.zeros((args.res, args.res, 3))
    lst = []
    for direction in range(3):
        for slicer in range(args.res):

            img_np[:, :, 0] = np.take(u, slicer, axis=direction)
            img_np[:, :, 1] = np.take(v, slicer, axis=direction)
            img_np[:, :, 2] = np.take(w, slicer, axis=direction)

            # Scale to between zero and one
            lbnd = np.min(img_np, axis=(0, 1))
            rbnd = np.max(img_np, axis=(0, 1))
            for k in range(3):
                img_np[:, :, k] = (img_np[:, :, k] - lbnd[k]) / (rbnd[k] - lbnd[k])

            # Save image and numpy array
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            sfx = "hit{0:04d}".format(cnt)
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
            cnt += 1

    scalings = pd.DataFrame(lst)
    scalings.to_csv(sname, index=False)

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
