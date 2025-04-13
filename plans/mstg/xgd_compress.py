import sys
from astropy.io.fits import open as fits_open, writeto as fits_save, getheader
from os import walk, path

# 压缩截取小光电文件，方便传输
def current_path():
    if getattr(sys, 'frozen', False):
        #base_path = sys._MEIPASS
        base_path = path.dirname(sys.executable)
    else:
        base_path = path.dirname(path.abspath(__file__))
    return base_path


def crop_fits_image(fits_file, size, cropped_file):
    """
    从给定的 FITS 文件中截取中心区域。

    参数：
    fits_file (str): FITS 文件的路径。
    size (int): 截取的区域大小（将截取 size x size 的区域）。
    cropped_file(str): 修改后存储的路径
    返回：
    numpy.ndarray: 截取的中心区域图像数据。
    """
    # 打开 FITS 文件
    try:
        with fits_open(fits_file) as hdul:
            # 假设图像数据在第一个扩展中
            image_data = hdul[0].data

            # 获取图像的形状
            height, width = image_data.shape

            # 计算中心区域的起始和结束坐标
            if width > size:
                start_x = (width - size) // 2
                end_x = start_x + size
            else:
                start_x = 0
                end_x = width
            if height > size:
                start_y = (height - size) // 2
                end_y = start_y + size
            else:
                start_y = 0
                end_y = height

            # 截取中心区域
            cropped_image = image_data[start_y:end_y, start_x:end_x]
            headers = hdul[0].header
            fits_save(cropped_file, cropped_image, header=headers, overwrite=True)  # update NAXIS
    except:
        cropped_image = None

    return cropped_image


# 示例用法
# fits_file_home = r'F:\etc\data2'  # FITS 文件根路径
crop_size = 1024  # 截取的区域大小
this_path = current_path()
fits_file_home = input(f'input fits file directory(default: {this_path}): ')
if not fits_file_home:
    fits_file_home = this_path
input_size = input(f'input crop size (default: {crop_size}): ')
crop_size = int(crop_size if not input_size else input_size)

for home, dirs, files in walk(fits_file_home):
    for file in files:
        if file.endswith('.fits') or file.endswith('.fit'):
            print(file)
            file_path = path.join(home, file)
            pos = -5 if file[-5] == '.' else -4
            cropped_file_path = path.join(fits_file_home, file[:pos] + "_cropped.fits.gz")
            cropped_image = crop_fits_image(file_path, crop_size, cropped_file_path)
            if cropped_image is None:
                print(f'{file} corrupt.')

print('done')
