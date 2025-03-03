import numpy as np
from PIL import Image

def remove_noise_threshold(image, threshold=50):
    image_array = np.array(image)
    
    image_array[image_array > threshold] = 255
    image_array[image_array <= threshold] = 0
    
    clean_image = Image.fromarray(image_array)
    return clean_image

image = Image.open('image2.png')
clean_image = remove_noise_threshold(image, threshold=50)

clean_image.save('threshold2.png')
clean_image.show()
