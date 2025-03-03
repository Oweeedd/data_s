from PIL import Image

frame_files1 = [f'frame1_{i}.png' for i in range(8)]
frame_files2 = [f'frame2_{i}.png' for i in range(8)]
frames1 = [Image.open(frame) for frame in frame_files1]
frames2 = [Image.open(frame) for frame in frame_files2]
total_width = sum(frame.width for frame in frames1)  
total_height = frames1[0].height  
combined_image1 = Image.new('L', (total_width, total_height))
combined_image2 = Image.new('L', (total_width, total_height))

x_offset = 0
for frame in frames1:
    combined_image1.paste(frame, (x_offset, 0))
    x_offset += frame.width

x_offset = 0
for frame in frames2:
    combined_image2.paste(frame, (x_offset, 0))
    x_offset += frame.width

combined_image1.save('combined_image1.png')
combined_image2.save('combined_image2.png')

