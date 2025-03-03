import numpy as np
from PIL import Image

def read_raw_data(file_path, width, height):
    
    with open(file_path, 'rb') as f:
        data = f.read()
    image = np.frombuffer(data, dtype=np.uint8)
    return image

total_width = 2048
total_height = 1280
num_frames = 8
frame_width = total_width // num_frames  
frame_height = total_height  

raw_data1 = read_raw_data('27012025-183000_lines_2048_points_1280.raw', total_width, total_height)
raw_data2 = read_raw_data('27012025-183006_lines_2048_points_1280.raw', total_width, total_height)

for i in range(num_frames):
    frame1 = raw_data1[i * frame_width * frame_height : (i + 1) * frame_width * frame_height]
    frame2 = raw_data2[i * frame_width * frame_height : (i + 1) * frame_width * frame_height]

    frame1 = frame1.reshape(frame_height, frame_width)
    frame2 = frame2.reshape(frame_height, frame_width)

    Image.fromarray(frame1).save(f'frame1_{i}.png')
    Image.fromarray(frame2).save(f'frame2_{i}.png')
