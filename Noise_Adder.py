# Import the libraries
import cv2
import skimage
from skimage import util

# Add Gaussian Noise to all the input images
for i in range(1,11):
    img=cv2.imread(r".\Data\Image"+str(i)+".png",0)/255 # Read the input image
    noisy_image=skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True)*255 # Add noise to the input image
    cv2.imwrite(r".\Gaussian_Noisy_Image\noisy_gauss_Image"+str(i)+".png",noisy_image) # Save the image in the respective folder

# Add Salt and Pepper Noise to all the input images
for i in range(1,11):
    img_1=cv2.imread(r".\Data\Image"+str(i)+".png",0)/255 # Read the input image
    noisy_image_1=skimage.util.random_noise(img_1, mode='s&p', seed=None, clip=True)*255 # Add noise to the input image
    cv2.imwrite(r".\Salt&Pepper_Noisy_Image\noisy_sp_Image"+str(i)+".png",noisy_image_1) # Save the image in the respective folder

print('Noise Addition Done') # Both the Gaussian and Salt & Pepper noise additions done