import csv
import os

import numpy as np
import pandas as pd
from PIL import Image


# 0 - Front
# 1 - Rear
# 2 - Side
# 3 - Front-Side
# 4 - Rear-Side

def createCSV_file_for_2(csv_path, image_path, txt_files_path):
    for root, dirs, files in os.walk(txt_files_path, topdown=True):
        header = ['filename', 'viewpoint']
        with open(csv_path, 'w', encoding='UTF8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for txt in files:
                txt_file = os.path.join(root, txt)
                img = image_path + '/' + txt[:-3] + 'jpg'
                with open(txt_file) as f:
                    viewpoint = f.readline()
                if viewpoint == 's':
                    if os.path.exists(img):
                        os.remove(img)
                    else:
                        print("The image does not exist")
                else:
                    writer.writerow([txt[:-3] + 'jpg', viewpoint])


def update_additional_front_rear(csv_path, front_image_path, rear_image_path):
    for root, dirs, files in os.walk(front_image_path, topdown=True):
        with open(csv_path, 'a', encoding='UTF8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for image in files:
                writer.writerow([image, '0'])

    for root, dirs, files in os.walk(rear_image_path, topdown=True):
        with open(csv_path, 'a', encoding='UTF8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for image in files:
                writer.writerow([image, '1'])


def spilt_train_test_query(csv_path, image_root, is_grayscale=False):
    train_path = os.path.join(image_root, 'image_train')
    test_path = os.path.join(image_root, 'image_test')
    query_path = os.path.join(image_root, 'image_query')
    train_csv_path = os.path.join(image_root, 'train_data.csv')
    df = pd.read_csv(csv_path)

    # train_test_query split
    train_df_mask = np.random.rand(len(df)) < 0.9
    train_df = df[train_df_mask]

    test_query_df = df[~train_df_mask]
    test_df_mask = np.random.rand(len(test_query_df)) < 0.7

    test_df = test_query_df[test_df_mask]
    query_df = test_query_df[~test_df_mask]

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
    if not os.path.isdir(query_path):
        os.mkdir(query_path)

    # saving imgs in split folders

    for train_img in train_df['filename']:
        img_path = os.path.join(image_root, train_img)
        dst_path = os.path.join(train_path, train_img)
        save_image(img_path, dst_path, is_grayscale)

    for test_img in test_df['filename']:
        img_path = os.path.join(image_root, test_img)
        dst_path = os.path.join(test_path, test_img)
        save_image(img_path, dst_path, is_grayscale)

    for query_img in query_df['filename']:
        img_path = os.path.join(image_root, query_img)
        dst_path = os.path.join(query_path, query_img)
        save_image(img_path, dst_path, is_grayscale)

    # creating new train csv
    header = ['filename', 'viewpoint']
    with open(train_csv_path, 'w', encoding='UTF8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for index in range(len(train_df)):
            image = train_df['filename'].iloc[index]
            viewpoint = train_df['viewpoint'].iloc[index]
            img_path = 'image_train/' + image[:-3] + 'jpg'
            writer.writerow([img_path, viewpoint])
def save_image(src_img, dst_path, is_grayscale=False):
    if is_grayscale:
        img = Image.open(src_img).convert('L')
        img.save(dst_path[:-3] + 'jpg')
    else:
        img = Image.open(src_img).convert('RGB')
        img.save(dst_path[:-3] + 'jpg')


if __name__ == '__main__':
    # csv_path = 'F:/Downloads/data/dataset.csv'
    # image_path = 'F:/Downloads/data/2/images'
    # txt_file_path = 'F:/Downloads/data/2/labels'
    # createCSV_file_for_2(csv_path, image_path, txt_file_path)

    # front_img_path = 'F:/Downloads/data/front_rear_images/front'
    # rear_img_path = 'F:/Downloads/data/front_rear_images/rear'
    # update_additional_front_rear(csv_path, front_img_path, rear_img_path)

    image_root = 'F:/Downloads/data/Mask_dataset/images'
    csv_path = 'F:/Downloads/data/Mask_dataset/dataset.csv'
    spilt_train_test_query(csv_path, image_root, is_grayscale=False)
    # https://dms.uom.lk/apps/files/?dir=/&fileid=7629451
