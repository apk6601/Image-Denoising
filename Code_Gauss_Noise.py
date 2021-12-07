# This code is for denoising of Gaussian Noisy images
# Import the libraries
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import sqrt

from PSNR_MSE import MSE,PSNR # Import the MSE and PSNR function

img_index=1 # Index of the image on which denoising is to be done (If input image name is 'Image3.png', enter 3 as img_index)

img_input = cv2.imread(r".\Data\Image"+str(img_index)+".png",0)/255 # Read the input image
noisy_img_input = cv2.imread(r".\Gaussian_Noisy_Image\noisy_gauss_Image"+str(img_index)+".png",0)/255 #Read the Noisy image

img_to_exp_gauss=noisy_img_input # Noisy image for Gaussian Filtering
img_to_exp_nlm=noisy_img_input # Noisy image for NL-Means denoising

# Gaussian Filtering Denoising

window_size=11 # The window size of the neighbourhood
sigma_gauss=0.8 # The sigma for the gaussian filtering

centre_x,centre_y=(window_size-1.0)/2.0,(window_size-1.0)/2.0 # Find the centre coordinates of the window
x,y = np.ogrid[-centre_x:centre_x+1, -centre_y:centre_y+1] # Create a grid
gauss = np.exp(-(x*x + y*y)/(2.0*sigma_gauss**2)) # The Gaussian distribution function

normalize=gauss/gauss.sum() # Normalize the gaussian distribution
gauss_den = signal.convolve2d(img_to_exp_gauss,normalize,mode='same') # Convolve the noisy image with the normalized gaussian distribution filter to obtain the denoised image



# Non Local Means Denoising

sigma_nlm=sqrt(2)*0.6 # The sigma for the NLM
window_l=10 # The window length for the neighbourhood
window_h=4 # The window width for the neighbourhood

h,w=img_to_exp_nlm.shape # The dimensions of the noisy image

img_pad=np.pad(img_to_exp_nlm,window_l+window_h,mode='constant',constant_values=0) # Add a pad of zeros around the noisy image matrix

Z=np.zeros(img_to_exp_nlm.shape) # The normalizing constant
weight=np.zeros(img_to_exp_nlm.shape) # The weights

# The Non-Local Means algorithm
for i in range(-window_l,window_l+1):
    for j in range(-window_l,window_l+1):

        power=np.zeros(img_to_exp_nlm.shape) # To store the weighted Euclidean distance

        for k in range(-window_h,window_h+1):
            for l in range(-window_h,window_h+1):

                # Calculating the weighted Euclidean distance
                store1=img_pad[i+k+window_l+window_h:h+i+k+window_l+window_h,j+l+window_l+window_h:w+j+l+window_l+window_h]
                store2=img_pad[k+window_l+window_h:h+k+window_l+window_h,l+window_l+window_h:w+l+window_l+window_h]

                power+=(store1-store2)**2 # The weighted Euclidean distance

        Z+=np.exp(-power/(sigma_nlm**2)) # Find the normalizing constant
        weight+=np.exp(-power/(sigma_nlm**2)) * img_pad[i+window_l+window_h:h+i+window_l+window_h,j+window_l+window_h:w+j+window_l+window_h] # Find the weights

nlm_den=weight/Z # The NLM denoised image

print('PSNR for Gaussian Filtering: ',PSNR(img_input,gauss_den)) # PSNR for Gaussian filtering
print('MSE for Gaussian Filtering: ',MSE(img_input,gauss_den)) # MSE for Gaussian filtering

print('PSNR for Non-Local Means: ',PSNR(img_input,nlm_den)) # PSNR for NLM
print('MSE for Non-Local Means: ',MSE(img_input,nlm_den)) # MSE for NLM

cv2.imwrite(r".\Gaussian_Noise_Denoised_Result\Gauss_Filter\Gauss_Filter_Result_Image"+str(img_index)+".png",gauss_den*255) # Save the Gaussian denoised image
cv2.imwrite(r".\Gaussian_Noise_Denoised_Result\Non_Local_Means\NLM_Result_Image"+str(img_index)+".png",nlm_den*255) # Save the NLM denoised image

plt.figure(figsize=(20,12)) # The plot window size

# Plot the input image
plt.subplot(2,2,1)
plt.imshow(img_input,cmap='gray')
plt.title("Input")

# Plot the noisy image
plt.subplot(2,2,2)
plt.imshow(noisy_img_input,cmap='gray')
plt.title("Noisy" + ": [PSNR {0:.3f} dB".format(PSNR(img_input, noisy_img_input)) + ", MSE {0:.4f}]".format(MSE(img_input, noisy_img_input)))

# Plot the Gaussian denoised image
plt.subplot(2,2,3)
plt.imshow(gauss_den,cmap='gray')
plt.title("Gaussian Filter" + ": [PSNR {0:.3f} dB".format(PSNR(img_input, gauss_den)) + ", MSE {0:.4f}]".format(MSE(img_input, gauss_den)))

# Plot the NLM denoised image
plt.subplot(2,2,4)
plt.imshow(nlm_den,cmap='gray')
plt.title("NL-Means" + ": [PSNR {0:.3f} dB".format(PSNR(img_input, nlm_den)) + ", MSE {0:.4f}]".format(MSE(img_input, nlm_den)))

plt.show() # Show the output