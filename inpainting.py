#!/usr/bin/env python3
"""Run through images and perform inpainting
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import argparse
import glob
import shutil
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import logging
import torch
import torch.optim
import scipy.io as sio

from deep_image_prior.models.resnet import ResNet
from deep_image_prior.models.unet import UNet
from deep_image_prior.models.skip import skip
import deep_image_prior.utils.inpainting_utils as utils


# ========================================================================
#
# Functions
#
# ========================================================================
def closure():

    global i

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    print("Iteration %05d    Loss %f" % (i, total_loss.item()), "\r", end="")
    if show_plot and i % show_every == 0:
        out_np = utils.torch_to_np(out)
        fname = os.path.join(odir, "out{0:05d}.png".format(i))
        out_pil = utils.np_to_pil(out_np)
        out_pil.save(fname)
        logger.info("Iteration %05d    Loss %f" % (i, total_loss.item()))

    i += 1

    return total_loss


# ========================================================================
def init_logger(fname):
    logger = logging.getLogger()

    # Clear any handlers that the logger may have
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=fname,
        filemode="w",
    )

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the handler to the root logger
    logger.addHandler(console)

    return logger


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # ========================================================================
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Perform inpainting using deep image priors"
    )
    parser.add_argument(
        "-r", "--resolution", dest="res", help="Image resolution", type=int, default=64
    )
    parser.add_argument(
        "-i",
        "--iterations",
        dest="iterations",
        help="Number of iterations",
        type=int,
        default=5001,
    )
    parser.add_argument(
        "--use_gpu",
        dest="use_gpu",
        help="Use a GPU to perform computations",
        action="store_true",
    )
    parser.add_argument(
        "--flow",
        dest="flow",
        help="Flow to be inpainted (turbulence | demo | cylinder | pipeflow)",
        type=str,
        default="turbulence",
    )
    parser.add_argument(
        "--ext",
        dest="ext",
        help="Extension of input format for data",
        type=str,
        default="png",
    )
    args = parser.parse_args()

    # ========================================================================
    # Setup
    seed = 89457364
    np.random.seed(seed)
    show_plot = True
    imgsize = -1

    if args.use_gpu:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Paths

    # Demo mode: just run the NREL mask case
    cases = {}
    cnt = 0
    if args.flow == "demo":
        root = os.path.abspath("turbulence")
        rdir = os.path.join(root, "results")
        imgdir = os.path.join(root, "images_{0:d}".format(args.res))
        maskdir = os.path.join(root, "masks_{0:d}".format(args.res))
        imgname = os.path.join(imgdir, "hit0000.{0:s}".format(args.ext))
        maskname = os.path.join(maskdir, "nrel.png")
        cases["demo{0:d}".format(args.res)] = {
            "imgname": imgname,
            "maskname": maskname,
            "mbox": -1,
        }
    # Homogeneous isotropic turbulence
    elif args.flow == "turbulence":
        nimages = 100
        root = os.path.abspath("turbulence")
        rdir = os.path.join(root, "results")
        imgdir = os.path.join(root, "images_{0:d}".format(args.res))
        maskdir = os.path.join(root, "masks_{0:d}".format(args.res))
        fname = "hit*.{0:s}".format(args.ext)
        imgnames = glob.glob(os.path.join(imgdir, fname))[:nimages]
        log = pd.read_csv(os.path.join(maskdir, "log.dat"))
        for imgname in imgnames:
            masks = log.sample(frac=1.0).groupby(by=["fraction", "nprocs"]).first()
            for index, row in masks.iterrows():
                cases["case{0:06d}".format(cnt)] = {
                    "imgname": imgname,
                    "maskname": os.path.join(maskdir, row.filename),
                    "mbox": row.mbox,
                }
                cnt += 1
    # Cylinder flow
    elif args.flow == "cylinder":
        root = os.path.abspath("cylinder")
        rdir = os.path.join(root, "results")
        imgdir = os.path.join(root, "images_{0:d}".format(args.res))
        maskdir = imgdir
        imgname = os.path.join(imgdir, "cyl.{0:s}".format(args.ext))
        log = pd.read_csv(os.path.join(maskdir, "log.dat"))
        for index, row in log.iterrows():
            cases["case{0:06d}".format(cnt)] = {
                "imgname": imgname,
                "maskname": os.path.join(maskdir, row.filename),
                "mbox": row.mbox,
            }
            cnt += 1
    # Pipeflow
    elif args.flow == "pipeflow":
        nimages = 147
        root = os.path.abspath("pipeflow")
        rdir = os.path.join(root, "results")
        imgdir = os.path.join(root, "images")
        maskdir = os.path.join(root, "masks")
        fname = "pf*.{0:s}".format(args.ext)
        imgnames = glob.glob(os.path.join(imgdir, fname))[:nimages]
        log = pd.read_csv(os.path.join(maskdir, "log.dat"))
        for imgname in imgnames:
            masks = log.sample(frac=1.0).groupby(by=["fraction", "nprocs"]).first()
            for index, row in masks.iterrows():
                cases["case{0:06d}".format(cnt)] = {
                    "imgname": imgname,
                    "maskname": os.path.join(maskdir, row.filename),
                    "mbox": row.mbox,
                }
                cnt += 1

    # ========================================================================
    # Loop on all cases
    print("Number of cases to run is", len(cases))
    for idx, case in cases.items():

        # Create directory to save results (and clean it)
        case_start = time.time()
        odir = os.path.join(rdir, idx)
        if os.path.exists(odir):
            for f in glob.glob(os.path.join(odir, "out*.png")):
                os.remove(f)
        else:
            os.makedirs(odir)

        # Pick image
        imgname = case["imgname"]
        maskname = case["maskname"]
        shutil.copyfile(maskname, os.path.join(odir, "mask.png"))

        # Load image, mask and center crop
        if os.path.splitext(imgname)[1] == ".npz":
            dat = np.load(imgname)
            img_np = dat["img"].transpose(2, 0, 1)
        elif os.path.splitext(imgname)[1] == ".mat":
            mat = sio.loadmat(imgname)
            img_np = mat["A"].transpose(2, 0, 1)
        else:
            _, img_np = utils.get_image(imgname, imgsize)
        img_pil = utils.np_to_pil(img_np)
        img_pil.save(os.path.join(odir, "image.png"))
        img_mask_pil, img_mask_np = utils.get_image(maskname, imgsize)

        # Log some stuff
        logname = os.path.join(odir, "run.log")
        logger = init_logger(logname)
        logger.info("Running " + idx)
        logger.info("Image name: {0:s}".format(imgname))
        logger.info("Mask name: {0:s}".format(maskname))
        missing = len(img_mask_np[0, :, :].flatten()) - int(
            np.sum(img_mask_np[0, :, :])
        )
        fraction = (1 - np.mean(img_mask_np[0, :, :])) * 100
        logger.info("Missing pixels: {0:d}".format(missing))
        logger.info("Percent image masked: {0:f}".format(fraction))
        logger.info("Individual mask box size: {0:f}".format(case["mbox"]))

        # Visualize masked image
        masked = utils.np_to_pil(img_mask_np * img_np)
        masked.save(os.path.join(odir, "masked.png"))

        # Setup learning
        pad = "reflection"  # "wrap" or "reflection" or "zero"
        if args.flow == "turbulence":
            pad = "wrap"
        OPT_OVER = "net"
        OPTIMIZER = "adam"

        INPUT = "meshgrid"
        input_depth = 2
        LR = 0.01
        num_iter = args.iterations
        param_noise = False
        show_every = 50
        figsize = 5
        reg_noise_std = 0.03
        net = skip(
            input_depth,
            img_np.shape[0],
            num_channels_down=[128] * 5,
            num_channels_up=[128] * 5,
            num_channels_skip=[0] * 5,
            upsample_mode="nearest",
            filter_skip_size=1,
            filter_size_up=3,
            filter_size_down=3,
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        ).type(dtype)

        net = net.type(dtype)
        net_input = utils.get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

        # Number of parameters
        nparams = sum(np.prod(list(p.size())) for p in net.parameters())
        logger.info("Number of parameters in neural network: {0:d}".format(nparams))

        # Loss
        mse = torch.nn.MSELoss().type(dtype)

        # Learning loop
        i = 0
        img_var = utils.np_to_torch(img_np).type(dtype)
        mask_var = utils.np_to_torch(img_mask_np).type(dtype)

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        p = utils.get_params(OPT_OVER, net, net_input)
        utils.optimize(OPTIMIZER, p, closure, LR, num_iter)

        out_np = utils.torch_to_np(net(net_input))

        # Only use the result where the mask was
        result = np.copy(img_np)
        result[img_mask_np < 0.5] = out_np[img_mask_np < 0.5]
        result_pil = utils.np_to_pil(result)
        result_pil.save(os.path.join(odir, "result.png"))

        # Save a file (numpy format)
        np.savez_compressed(
            os.path.join(odir, "data.npz"),
            out=out_np,
            result=result,
            original=img_np,
            mask=img_mask_np,
            mbox=case["mbox"],
            fraction=fraction,
        )

        end = time.time() - case_start
        logger.info(
            "Elapsed time "
            + str(timedelta(seconds=end))
            + " (or {0:f} seconds)".format(end)
        )

    end = time.time() - start
    print(
        "Elapsed total time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
