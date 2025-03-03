import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# Чтение
def read_raw_image(file_path, width=2048, height=1280, byte_order='little', flip_vertically=False, flip_horizontally=False):
    file_size = os.path.getsize(file_path)
    expected_size_8bit = width * height

    print(f"\nПроверка файла: {file_path}")
    print(f"Размер файла: {file_size} байт")
    print(f"Для 8-бит: {expected_size_8bit} байт")

    if file_size == expected_size_8bit:
        dtype = np.uint8

    else:
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

#  Нормализация 
def normalize_image(img, target_dtype=np.uint8):
    if img.dtype != target_dtype:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(target_dtype)
    return img

# Сравнение
def compare_images(img1, img2):
    if img1.shape != img2.shape:
        print(f"Размеры изображений не совпадают: img1 {img1.shape}, img2 {img2.shape}")
        img2 = cv2.transpose(img2)
        print(f"img2 после транспонирования: {img2.shape}")
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            print(f"Ресайз img2 до: {img2.shape}")

    if img1.dtype != img2.dtype:
        img2 = img2.astype(img1.dtype)

    diff = cv2.absdiff(img1, img2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    filtered = cv2.medianBlur(thresh, 5)
    return diff, filtered, img2

# Анализ различий
def analyze_difference(img1, img2, diff):
    differing_pixels = np.sum(diff != 0)
    total_pixels = diff.size
    diff_percentage = (differing_pixels / total_pixels) * 100
    mse = np.mean((img1 - img2) ** 2)
    ssim_index = ssim(img1, img2)
    corr_coef, _ = pearsonr(img1.ravel(), img2.ravel())

    print(f"\n--- Анализ различий ---")
    print(f"Изменённых пикселей: {differing_pixels}/{total_pixels} ({diff_percentage:.2f}%)")
    print(f"MSE (Среднеквадратичная ошибка): {mse:.2f}")
    print(f"SSIM (Индекс структурного сходства): {ssim_index:.4f}")
    print(f"Коэффициент корреляции гистограмм: {corr_coef:.4f}")

# Визуализация
def plot_images(img1, img2, diff, filtered, title1="Изображение 1", title2="Изображение 2"):
    overlay = cv2.addWeighted(img1, 0.7, diff, 0.3, 0)

    plt.figure(figsize=(18, 8))

    plt.subplot(2, 3, 1)
    plt.title(title1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(title2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Разница")
    plt.imshow(diff, cmap='hot')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Фильтрованная разница")
    plt.imshow(filtered, cmap='hot')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Наложение разницы")
    plt.imshow(overlay, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Гистограммы
def plot_histograms(img1, img2, title1="RAW 1", title2="RAW 2"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Гистограмма {title1}")
    plt.hist(img1.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.title(f"Гистограмма {title2}")
    plt.hist(img2.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)

    plt.tight_layout()
    plt.show()

    cdf1 = np.cumsum(np.histogram(img1.ravel(), bins=256, range=(0, 256))[0])
    cdf2 = np.cumsum(np.histogram(img2.ravel(), bins=256, range=(0, 256))[0])

    plt.figure(figsize=(12, 5))
    plt.title("Кумулятивные гистограммы (CDF)")
    plt.plot(cdf1, color='blue', label=title1)
    plt.plot(cdf2, color='green', label=title2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    raw_file1 = r'27012025-183000_lines_2048_points_1280.raw'
    raw_file2 = r'27012025-183006_lines_2048_points_1280.raw'

   
    raw_img1 = read_raw_image(raw_file1, byte_order='little', flip_vertically=False)
    raw_img2 = read_raw_image(raw_file2, byte_order='little', flip_vertically=False)

   
    norm_raw_img1 = normalize_image(raw_img1)
    norm_raw_img2 = normalize_image(raw_img2)

   
    diff_raw, filtered_raw, aligned_raw2 = compare_images(norm_raw_img1, norm_raw_img2)
    analyze_difference(norm_raw_img1, aligned_raw2, diff_raw)
    plot_images(norm_raw_img1, aligned_raw2, diff_raw, filtered_raw, "RAW 1", "RAW 2")
    plot_histograms(norm_raw_img1, aligned_raw2, "RAW 1", "RAW 2")
