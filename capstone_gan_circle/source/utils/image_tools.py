import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_image_similarity(input_image, enhanced_image):
    L = input_image.max() - input_image.min()

    sim = ssim(input_image, enhanced_image, gaussian_weights=True,
               sigma=1.5, win_size=11, data_range=L)
    peak = psnr(input_image, enhanced_image, data_range=L)
    ambe = np.abs(np.mean(input_image)-np.mean(enhanced_image)) / L

    return sim, peak, ambe
