#!/usr/bin/env python3
"""Generate some image masks
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

# from skimage.draw import random_shapes


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate masks")
    parser.add_argument(
        "-r", "--resolution", dest="res", help="Image resolution", type=int, default=64
    )
    args = parser.parse_args()

    # Setup
    seed = 89457364
    np.random.seed(seed)
    tpath = os.path.abspath("turbulence")
    odir = os.path.abspath(os.path.join(tpath, "masks_{0:d}".format(args.res)))
    lname = os.path.abspath(os.path.join(odir, "log.dat"))
    if os.path.exists(odir):
        for f in glob.glob(os.path.join(odir, "mask*.png")):
            os.remove(f)
    else:
        os.makedirs(odir)

    # Sweep number of processors and number of dead procs
    cases = {
        2: [1. / 4],
        4: [1. / 16, 1. / 8, 1. / 4],
        8: [1. / 16, 1. / 8, 1. / 4],
        16: [1. / 16, 1. / 8, 1. / 4],
        32: [1. / 16, 1. / 8, 1. / 4],
    }
    nsamples = 10
    cnt = 0
    lst = []
    for nprocs_x, fractions in cases.items():
        for fraction in fractions:
            for nsample in range(nsamples):

                # Randomly select some processors to be unavailable
                nprocs = int(nprocs_x ** 2)
                ndead = int(nprocs * fraction)
                arr = np.array([0] * ndead + [1] * (nprocs - ndead))
                np.random.shuffle(arr)
                pmask = arr.reshape(nprocs_x, nprocs_x)

                # Generate the mask from these unavailable procs
                mbox = args.res // nprocs_x
                mask = np.kron(pmask, np.ones((mbox, mbox)))
                img_np = (np.repeat(mask[:, :, np.newaxis], 3, axis=2) * 255).astype(
                    np.uint8
                )

                # # Generate the random mask with random shapes
                # fraction_percent = 10
                # nshapes = 4
                # msize = int(np.sqrt(fraction_percent / 100 * args.res ** 2 / nshapes))
                # mask, labels = random_shapes(
                #     (args.res, args.res),
                #     max_shapes=nshapes,
                #     min_shapes=nshapes,
                #     min_size=msize,
                #     max_size=msize,
                #     allow_overlap=False,
                #     shape="rectangle",
                #     multichannel=False,
                #     intensity_range=(0, 0),
                # )
                # img_np = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                actual = (1 - np.mean(img_np[:, :, 0]) / 255) * 100
                print(
                    "Percent of image blacked out = {0:.2f} (target = {1:.2f})".format(
                        actual, fraction * 100
                    )
                )

                # Save the image
                img_pil = Image.fromarray(img_np)
                fname = "mask{0:04d}.png".format(cnt)
                img_pil.save(os.path.join(odir, fname))
                lst.append(
                    {
                        "nprocs": nprocs_x,
                        "ndead": ndead,
                        "mbox": mbox,
                        "fraction": fraction,
                        "filename": fname,
                    }
                )
                cnt = cnt + 1

    log = pd.DataFrame(lst)
    log.to_csv(lname, index=False)

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
