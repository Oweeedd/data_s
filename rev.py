import numpy as np
from PIL import Image, ImageChops

def raw_to_image(file_path, width, height):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    
    assert data.size == width * height, f"Размер  не совпадает {data.size} != {width * height}"
    
    image = data.reshape((height, width))
    
    img = Image.fromarray(image)
    return img

def compare_images(image1, image2):
    diff = ImageChops.difference(image1, image2)
    diff = diff.convert('L')
    diff = diff.point(lambda p: p > 10 and 255)
    diff.save('difference.png')
    diff.show()

width, height = 2048, 1280  
raw_file1_path = '27012025-183000_lines_2048_points_1280.raw'
image1 = raw_to_image(raw_file1_path, width, height)
raw_file2_path = '27012025-183006_lines_2048_points_1280.raw'
image2 = raw_to_image(raw_file2_path, width, height)
compare_images(image1, image2)
