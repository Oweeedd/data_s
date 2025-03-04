import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.fft import fft2, fftshift
from scipy.signal import wiener

# Чтение данных
def read_raw_image(file_path, width=2048, height=1280, dtype=np.uint8, byte_order='little', flip_vertically=False, flip_horizontally=False):
    file_size = os.path.getsize(file_path)
    expected_size = width * height * np.dtype(dtype).itemsize

    print(f"\nПроверка файла: {file_path}")
    print(f"Размер файла: {file_size} байт")
    print(f"Ожидаемый размер: {expected_size} байт")

    if file_size != expected_size:
        raise ValueError("Размер не совпадает")

    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=dtype)

    if byte_order == 'big':
        raw_data.byteswap(inplace=True)

    image = raw_data.reshape((height, width))

    if flip_vertically:
        image = np.flipud(image)
    if flip_horizontally:
        image = np.fliplr(image)

    plt.figure(figsize=(6, 4))
    plt.title(f"Предварительный просмотр: {os.path.basename(file_path)}")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    return image

# Нормализация
def normalize_image(img, target_dtype=np.uint8):
    if img.dtype != target_dtype:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(target_dtype)
    return img

def apply_filters(image):
    # Медианный фильтр
    median_filtered = cv2.medianBlur(image, 5)
    
    # Гауссовский фильтр
    gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)
    
    # Wiener
    wiener_filtered = wiener(gaussian_filtered, mysize=5)
    wiener_filtered = np.uint8(cv2.normalize(wiener_filtered, None, 0, 255, cv2.NORM_MINMAX))
    
    return median_filtered, gaussian_filtered, wiener_filtered

def apply_threshold(image, threshold_value=30):
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

# Фурье
def apply_fourier_transform(image):
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shifted) + 1)  
    return magnitude_spectrum

# Визуализация
def plot_results(original, median, gaussian, wiener, thresholded, fourier):
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.title("Оригинал")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Медианный фильтр")
    plt.imshow(median, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Гауссовский фильтр")
    plt.imshow(gaussian, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Wiener фильтр")
    plt.imshow(wiener, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Пороговая обработка")
    plt.imshow(thresholded, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Преобразование Фурье")
    plt.imshow(fourier, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Основной код
if __name__ == '__main__':
    raw_file1 = r'1/27012025-183000_lines_2048_points_1280.raw'
    raw_file2 = r'1/27012025-183006_lines_2048_points_1280.raw'

    # Чтение данных
    raw_img1 = read_raw_image(raw_file1, dtype=np.uint8, byte_order='little', flip_vertically=False)
    raw_img2 = read_raw_image(raw_file2, dtype=np.uint8, byte_order='little', flip_vertically=False)

    # Нормализация
    norm_raw_img1 = normalize_image(raw_img1)
    norm_raw_img2 = normalize_image(raw_img2)

    # Применение фильтров
    median1, gaussian1, wiener1 = apply_filters(norm_raw_img1)
    median2, gaussian2, wiener2 = apply_filters(norm_raw_img2)

    # Пороговая обработка
    thresholded1 = apply_threshold(wiener1)
    thresholded2 = apply_threshold(wiener2)

    # Преобразование Фурье
    fourier1 = apply_fourier_transform(norm_raw_img1)
    fourier2 = apply_fourier_transform(norm_raw_img2)

    # Визуализация результатов
    plot_results(norm_raw_img1, median1, gaussian1, wiener1, thresholded1, fourier1)
    plot_results(norm_raw_img2, median2, gaussian2, wiener2, thresholded2, fourier2)