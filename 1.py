import numpy as np
from PIL import Image

def raw_to_image(file_path, width, height):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    
    assert data.size == width * height, f"Размер не совпадает {data.size} != {width * height}"
    image = data.reshape((height, width))
    img = Image.fromarray(image)
    return img

raw_file_path = '27012025-183006_lines_2048_points_1280.raw'
width, height = 2048, 1280  
image = raw_to_image(raw_file_path, width, height)

image.save('output_image2.png')
image.show()
