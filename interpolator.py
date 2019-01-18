#!/usr/bin/env python3
"""Inpainting through interpolation
"""
# ========================================================================
#
# Imports
#
# ========================================================================
import os
import argparse
import time
from datetime import timedelta
import numpy as np
from scipy import interpolate
from mpi4py import MPI
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import gpytorch
import deep_image_prior.utils.inpainting_utils as utils


# ========================================================================
#
# Functions
#
# ========================================================================
def interplation(img, mask, kind="linear"):
    """Interpolate img at the mask values.

    :param img: image (C x H x W)
    :type img: array 
    :param mask: mask (C x H x W)
    :type mask: array
    :param kind: type of interpolation
    :type kind: str
    :return: interpolated image (C x H x W)
    :rtype: array
    """

    interpolated = np.zeros(img.shape)
    for c in range(img.shape[0]):
        z = np.ma.array(
            np.squeeze(img[c, :, :]).T, mask=np.squeeze(mask[c, :, :] < 0.5).T
        )
        x, y = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
        x1 = x[~z.mask]
        y1 = y[~z.mask]
        z1 = z[~z.mask]
        interpolated[c, :, :] = interpolate.interp2d(x1, y1, z1, kind=kind)(
            np.arange(z.shape[0]), np.arange(z.shape[1])
        )
    return interpolated


# ========================================================================
def gpr(img, mask):
    """Interpolate img at the mask values using Gaussian Process Regression

    Uses standard GPR and therefore uses only pixels surrounding the mask

    :param img: image (C x H x W)
    :type img: array
    :param mask: mask (C x H x W)
    :type mask: array
    :return: interpolated image (C x H x W)
    :rtype: array
    """

    gpr = np.copy(img)
    x = np.linspace(0, 1, img.shape[2])
    y = np.linspace(0, 1, img.shape[1])
    xx, yy = np.meshgrid(x, y)
    midx = mask[0, :, :] > 0.5
    border = np.copy(midx)

    # Get pixels surrounding the mask. Hacky way of doing it but it works well.
    # original mask:         [T,T,T,F,F,F,F,T,T,T]
    # grown mask by 1 pixel: [T,T,F,F,F,F,F,F,T,T]
    # XOR for the border:    [F,F,T,F,F,F,F,T,F,F]
    n_border = 10
    for cnt in range(n_border):
        tmp = np.copy(border)
        for i in range(img.shape[1] - 1):
            for j in range(img.shape[2] - 1):
                if (tmp[i, j]) and (not tmp[i + 1, j]):
                    border[i, j] = False
                if (not tmp[i, j]) and (tmp[i + 1, j]):
                    border[i + 1, j] = False
                if (tmp[i, j]) and (not tmp[i, j + 1]):
                    border[i, j] = False
                if (not tmp[i, j]) and (tmp[i, j + 1]):
                    border[i, j + 1] = False
                if (tmp[i, j]) and (not tmp[i + 1, j + 1]):
                    border[i, j] = False
                if (tmp[i + 1, j]) and (not tmp[i, j + 1]):
                    border[i + 1, j] = False
                if (tmp[i, j + 1]) and (not tmp[i + 1, j]):
                    border[i, j + 1] = False
                if (tmp[i + 1, j + 1]) and (not tmp[i, j]):
                    border[i + 1, j + 1] = False
    idx = np.logical_xor(midx, border)

    # Loop on channels and use values around the mask to train the GP
    midx = mask[0, :, :] < 0.5
    gpr[:, midx] = 0
    for c in range(img.shape[0]):
        gp = GaussianProcessRegressor()
        gp.fit(X=np.column_stack([xx[idx], yy[idx]]), y=img[c, idx])
        rr_cc_as_cols = np.column_stack([xx[midx].flatten(), yy[midx].flatten()])
        gpr[c, midx] = gp.predict(rr_cc_as_cols)

    return gpr


# ========================================================================
def gpr_gpytorch(img, mask, use_gpu=False):
    """Interpolate img at the mask values using Gaussian Process Regression

    This uses the KISS-GP (Kernel interpolation for scalable
    structured Gaussian processes) framework introduced here
    http://proceedings.mlr.press/v37/wilson15.pdf. More information
    can be found at
    https://gpytorch.readthedocs.io/en/latest/examples/05_Scalable_GP_Regression_Multidimensional/KISSGP_Kronecker_Regression.html

    :param img: image (C x H x W)
    :type img: array
    :param mask: mask (C x H x W)
    :type mask: array
    :return: interpolated image (C x H x W)
    :rtype: array

    """

    # gpytorch specific parameters
    training_iter = 100
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float
    torch.manual_seed(5465462)

    # Other setup
    x = np.linspace(0, img.shape[2], img.shape[2])
    y = np.linspace(0, img.shape[1], img.shape[1])
    xx, yy = np.meshgrid(x, y)
    gpr = np.copy(img)
    midx = mask[0, :, :] < 0.5
    gpr[:, midx] = 0

    # Loop on channels
    for c in range(img.shape[0]):

        # Training data is outside the mask, dev data is inside the mask
        Xtrain = torch.as_tensor(
            np.column_stack([xx[~midx].flatten(), yy[~midx].flatten()]),
            dtype=dtype,
            device=device,
        )
        Xdev = torch.as_tensor(
            np.column_stack([xx[midx].flatten(), yy[midx].flatten()]),
            dtype=dtype,
            device=device,
        )

        # Target values
        Ytrain = torch.as_tensor(img[c, ~midx].flatten(), dtype=dtype, device=device)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(Xtrain, Ytrain, likelihood).to(device=device)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}],  # Includes GaussianLikelihood parameters
            lr=0.1,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            output = model(Xtrain)

            # Calc loss and backprop gradients
            loss = -mll(output, Ytrain)
            loss.backward()
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iter, loss.item()))
            optimizer.step()

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            observed_pred = likelihood(model(Xdev))
            gpr[c, midx] = observed_pred.mean.cpu().numpy()

    return gpr


# ========================================================================
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help
        # with that. Here we are using a grid that has the same number
        # of points as the training data (a ratio of 1.0). Performance
        # can be sensitive to this parameter, so you may want to
        # adjust it for your own problem on a validation set.
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
            ),
            grid_size=grid_size,
            num_dims=train_x.shape[-1],
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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
        description="Perform inpainting using interpolation"
    )
    parser.add_argument(
        "-k",
        "--kind",
        dest="kind",
        help="Interpolation kind (linear, cubic, quintic, gp, gpytorch)",
        type=str,
        default="gp",
    )
    parser.add_argument(
        "-r",
        "--results_directory",
        dest="results_directory",
        help="Directory containing the results to interpolate",
        type=str,
        default="cylinder/results",
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        help="Boundary condition (wrap)",
        type=str,
        default="",
    )
    parser.add_argument(
        "--use_gpu",
        dest="use_gpu",
        help="Use a GPU to perform computations",
        action="store_true",
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
    for cnt, fdir in enumerate(lfdirs):

        # Load the data
        print("Interpolating {0:s}".format(fdir))
        fname = os.path.join(fdir, "data.npz")
        data = np.load(fname)
        img = data["original"]
        mask = data["mask"]

        # Set boundary conditions by padding the image
        if args.mode == "wrap":
            nx, ny = img.shape[2], img.shape[1]
            img = np.tile(img, (3, 3))
            tmp = np.ones(img.shape)
            tmp[:, nx : 2 * nx, ny : 2 * ny] = mask
            mask = np.copy(tmp)

        # Interpolation
        if args.kind == "gp":
            interp_np = gpr(img, mask)
        elif args.kind == "gpytorch":
            interp_np = gpr_gpytorch(img, mask, args.use_gpu)
        else:
            interp_np = interplation(img, mask, kind=args.kind)

        # Take only the interior for periodic bc (discard padding)
        if args.mode == "wrap":
            interp_np = interp_np[:, nx : 2 * nx, ny : 2 * ny]

        # Save the image and array
        opfx = os.path.join(fdir, "interpolated")
        interp_pil = utils.np_to_pil(interp_np)
        interp_pil.save(opfx + ".png")
        np.savez_compressed(opfx + ".npz", interpolated=interp_np)

    comm.Barrier()
    if rank == 0:
        end = time.time() - start
        print(
            "Elapsed total time "
            + str(timedelta(seconds=end))
            + " (or {0:f} seconds)".format(end)
        )
