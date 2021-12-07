# Importing the libraries
import numpy as np
from math import log10, sqrt

# The Mean-Square Error function
def MSE(original,denoised):
    mse = np.mean((original - denoised) ** 2) # MSE
    return mse

# The PSNR function
def PSNR(original,denoised):
    mse=MSE(original,denoised) # Return the MSE
    if(mse==0): # If MSE=0, it means there is no noise, so psnr will be 100
        return 100

    psnr = 20 * log10(1.0 / sqrt(mse)) # PSNR
    return psnr
