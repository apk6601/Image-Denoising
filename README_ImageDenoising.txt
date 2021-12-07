The submission contains the following code files:
a) Noise_Adder: This file adds 'Gaussian' and 'Salt and Pepper' noises to the input image files and stores them in respective folders.
b) PSNR_MSE: This file is used to calculate the PSNR and MSE for the denoised versions of the input images.
c) Code_Gauss_Noise: This file is used to find the denoised versions of the Gaussian filter and the NL-means algorithm for Gaussian Noise images.
d) Code_SAP_Noise: This file is used to find the denoised versions of the Gaussian filter and the NL-means algorithm for Salt and Pepper Noise images.


The submission contains the following files, folders and subfolders:
a) Gaussian_Noisy_Image: This folder contains the input images with Gaussian noise added to them.
b) Salt&Pepper_Noisy_Image: This folder contains the input images with Salt and Pepper noise added to them.
c) Gaussian_Noise_Denoised_Result: This folder contains the denoised images of the Gaussian noisy images. The Gaussian Filter results are stored in 'Gauss_Filter' subfolder and the NLM results are stored in 'Non_Local_Means' subfolder.
d) SAP_Noise_Denoised_Result: This folder contains the denoised images of the Salt and Pepper noisy images. The Gaussian Filter results are stored in 'Gauss_Filter' subfolder and the NLM results are stored in 'Non_Local_Means' subfolder.
e) Data: This folder contains the input images.
f) README: This is the README file.


Steps to run the code:
a) Run the 'Noise_Adder.py' file to add the Gaussian Noise to the input images (these will be stored in 'Gaussian_Noisy_Image' folder) and the Salt and Pepper Noise to the input images (these will be stored in 'Salt&Pepper_Noisy_Image' folder).
b) Run the 'Code_Gauss_Noise.py' file to obtain the Gaussian Filter and NLM denoised images of the Gaussian noisy images.
c) The results will be stored in 'Gaussian_Noise_Denoised_Result' folder ('Gauss_Filter' subfolder for Gaussian filter denoised images and 'Non_Local_Means' subfolder for NLM denoised images).
d) Run the 'Code_SAP_Noise.py' file to obtain the Gaussian Filter and NLM denoised images of the Salt and Pepper noisy images.
e) The results will be stored in 'SAP_Noise_Denoised_Result' folder ('Gauss_Filter' subfolder for Gaussian filter denoised images and 'Non_Local_Means' subfolder for NLM denoised images).
f) In order to see output for a different image, change the value of 'img_index' variable in Line 11 (for both codes).
g) All the four images: Input image, Noisy image, Gaussian Filter denoised image, NLM denoised image will appear in the output window.