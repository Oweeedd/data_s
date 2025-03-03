from PIL import Image
import numpy as np
from scipy.ndimage import median_filter

def remove_noise_median_filter(image, size=3):
    image_array = np.array(image)
    
    filtered_image_array = median_filter(image_array, size=size)
    
    clean_image = Image.fromarray(filtered_image_array)
    return clean_image

image = Image.open('output_image2.png')
clean_image = remove_noise_median_filter(image, size=3)
clean_image.save('median2.png')
clean_image.show()
