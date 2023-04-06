import os

import cv2
from PIL import Image


def save_image(src_img, dst_path, is_grayscale=False, delete_src=False):
    if is_grayscale:
        img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
        converted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(dst_path[:-3] + 'jpg', converted_img)
    else:
        img = Image.open(src_img).convert('RGB')
        img.save(dst_path[:-3] + 'jpg')

    if delete_src:
        os.remove(src_img)


def png_to_jpg(data_path):
    for root, dirs, files in os.walk(data_path, topdown=True):
        if len(dirs) == 0:
            for img in files:
                src = os.path.join(root, img)
                dst = src
                save_image(src, dst, is_grayscale=False, delete_src=True)


def rename_folder_names(data_path):
    first = True
    for root, dirs, files in os.walk(data_path, topdown=True):
        if first:
            first = False
        else:
            identities = dirs
            print(identities)
            for i, boat in enumerate(identities):
                num = f'{i:02d}'
                os.rename(os.path.join(root, boat), os.path.join(root, num))

if __name__ == '__main__':
    print('##Converting png to jpg##')
    data_path = 'F:/Downloads/archive/Ships dataset/ReID_RGB'
    png_to_jpg(data_path)
    print('##renaming##')
    rename_folder_names(data_path)