import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.metrics import structural_similarity as ssim

def read_raw_image(file_path, width=2048, height=1280):
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)
    return raw_data.reshape((height, width))

# Применение медианного фильтра и размытия по Гауссу для удаления шума
def denoise_image(image):
    image = cv2.medianBlur(image, 1)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

# Наложение двух изображений, сохраняя белые пиксели сверху
def overlay_images(image1, image2):
    """."""
    return np.maximum(image1, image2)

# Сравнение двух RAW изображений и вывод статистики различий
def compare_raw_images(image1, image2):
    total_pixels = image1.size
    differing_pixels = np.sum(image1 != image2)
    matching_pixels = total_pixels - differing_pixels
    match_percentage = (matching_pixels / total_pixels) * 100
    diff_percentage = (differing_pixels / total_pixels) * 100
    
    mse_value = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    ssim_value = ssim(image1, image2, data_range=image2.max() - image2.min())
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    print(f"Совпадающих пикселей: {matching_pixels}/{total_pixels} ({match_percentage:.2f}%)")
    print(f"Отличающихся пикселей: {differing_pixels}/{total_pixels} ({diff_percentage:.2f}%)")
    print(f"MSE (Среднеквадратичная ошибка): {mse_value:.4f}")
    print(f"SSIM (Индекс структурного сходства): {ssim_value:.4f}")
    print(f"Коэффициент корреляции гистограмм: {hist_corr:.4f}")


#Анализ направления движения между двумя изображениями
def detect_motion(image1, image2):
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = image1.shape
    motion_image = np.zeros((h, w, 3), dtype=np.uint8)
    step = 30
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x] * 10
            cv2.arrowedLine(motion_image, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 2, tipLength=0.5)
    return motion_image


# Обработка двух изображений и вывод результатов
def process_radar_images(file1, file2):
    raw_image1 = read_raw_image(file1)
    raw_image2 = read_raw_image(file2)
    
    print(f"Проверка файла: {file1}")
    print(f"Размер файла: {os.path.getsize(file1)} байт")
    print(f"Проверка файла: {file2}")
    print(f"Размер файла: {os.path.getsize(file2)} байт")
    
    compare_raw_images(raw_image1, raw_image2)
    
    denoised_image1 = denoise_image(raw_image1)
    denoised_image2 = denoise_image(raw_image2)
    overlay_image = overlay_images(denoised_image1, denoised_image2)
    motion_image = detect_motion(denoised_image1, denoised_image2)
    
    return denoised_image1, denoised_image2, overlay_image, motion_image


raw_file1 = r'1/27012025-183000_lines_2048_points_1280.raw'
raw_file2 = r'1/27012025-183006_lines_2048_points_1280.raw'

denoised_image1, denoised_image2, overlay_image, motion_image = process_radar_images(raw_file1, raw_file2)

# Визуализация изображений
plt.figure(figsize=(16, 8))  

plt.subplot(1, 4, 1)
plt.imshow(denoised_image1, cmap='gray')
plt.title("Фильтрованное изображение 1")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(denoised_image2, cmap='gray')
plt.title("Фильтрованное изображение 2")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(overlay_image, cmap='gray')
plt.title("Наложенное изображение")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(motion_image)
plt.title("Анализ направления движения")
plt.axis('off')

plt.tight_layout()
plt.show()

output_dir = os.path.dirname(raw_file1)  
overlay_output_path = os.path.join(output_dir, "overlay_image.png")
motion_output_path = os.path.join(output_dir, "motion_image.png")

cv2.imwrite(overlay_output_path, overlay_image)
cv2.imwrite(motion_output_path, motion_image)

print(f"Изображение разницы сохранено в: {overlay_output_path}")
print(f"Изображение анализа направления движения сохранено в: {motion_output_path}")