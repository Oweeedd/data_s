import numpy as np
import os

def read_raw_image(file_path, width=2048, height=1280, byte_order='little', flip_vertically=False, flip_horizontally=False):
    file_size = os.path.getsize(file_path)
    expected_size_8bit = width * height
    expected_size_16bit = expected_size_8bit * 2

    print(f"\nПроверка файла: {file_path}")
    print(f"Размер файла: {file_size} байт")
    print(f"Ожидаемый размер для 8-бит: {expected_size_8bit} байт")
    print(f"Ожидаемый размер для 16-бит: {expected_size_16bit} байт\n")

    if file_size == expected_size_8bit:
        dtype = np.uint8
    elif file_size == expected_size_16bit:
        dtype = np.uint16
    else:
        raise ValueError("Размер файла не совпадает с ожидаемым. Проверьте формат или размер изображения.")

    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=dtype)

    if byte_order == 'big':
        raw_data.byteswap(inplace=True)

    image = raw_data.reshape((height, width))

    if flip_vertically:
        image = np.flipud(image)
    if flip_horizontally:
        image = np.fliplr(image)

    return image

def raw_to_arff(raw_file1, raw_file2, arff_file, width=2048, height=1280):
    # Чтение .raw файлов
    image1 = read_raw_image(raw_file1, width, height)
    image2 = read_raw_image(raw_file2, width, height)

    # Преобразование изображений в одномерные массивы
    data1 = image1.ravel()
    data2 = image2.ravel()

    # Создание заголовка .arff файла
    arff_header = f"""@RELATION image_comparison

@ATTRIBUTE pixel_value1 NUMERIC
@ATTRIBUTE pixel_value2 NUMERIC
@ATTRIBUTE image_id {{1, 2}}

@DATA
"""

    # Запись данных в .arff файл
    with open(arff_file, 'w') as f:
        f.write(arff_header)
        for value1, value2 in zip(data1, data2):
            f.write(f"{value1},{value2},1\n")  # Данные из первого изображения
            f.write(f"{value1},{value2},2\n")  # Данные из второго изображения

    print(f"Файл {arff_file} успешно создан.")

# Пример использования
if __name__ == '__main__':
    raw_file1 = r'27012025-183000_lines_2048_points_1280.raw'
    raw_file2 = r'27012025-183006_lines_2048_points_1280.raw'

    arff_file = 'image_comparison.arff'

    # Преобразование .raw в .arff
    raw_to_arff(raw_file1, raw_file2, arff_file)