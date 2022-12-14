import os
import random
from shutil import copyfile, rmtree
from distutils.dir_util import copy_tree
from PIL import Image

# Helper function to save image as grayscale
def save_grayscale(source, destination):
    img = Image.open(source).convert('L')
    img.save(destination)


# Checks the boat class from the labels and return the class as a string
def boat_class(txt_file_path):
    file = open(txt_file_path, 'r')
    contents = file.read()
    boat_cls = contents.split(' ')
    return boat_cls[0]


def clean_ID(name):
    ID = name.split('_')[0]
    slices = ID.split('-')
    if slices[-1].isnumeric():
        clean_name = '-'.join(slices[:-1])
    else:
        clean_name = '-'.join(slices)
    return clean_name


# You only need to change this line to your dataset download path

download_path = '/home/fyp3-2/Desktop/BATCH18/archive/Ships dataset'


if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# ---------------------------------------
# train
# This takes all the images from train and val and creates train folder
train_path = download_path + '/train/images'
train_label_path = download_path + '/train/labels'
val_path = download_path + '/val/images'
val_label_path = download_path + '/val/labels'
test_path = download_path + '/test/images'
test_label_path = download_path + '/test/labels'
train_save_path = download_path + '/pytorch/train'

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    for paths in [(train_path, train_label_path), (val_path, val_label_path), (test_path, test_label_path)]:
        for root, dirs, files in os.walk(paths[0], topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                ID = clean_ID(name)
                src_path = paths[0] + '/' + name
                src_label_path = paths[1] + '/' + name[:-3] + 'txt'
                dst_path = train_save_path + '/' + ID + boat_class(src_label_path)
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                save_grayscale(src_path, dst_path + '/' + name[:-3]+'jpg')
    # rmtree(train_save_path + '/' + 'USS-Bulkeley-DDG845')
iteration = -1
folders = ()
for root, dirs, files in os.walk(train_save_path, topdown=True):
    if iteration == -1:
        folders = dirs
    else:
        if len(files) < 5:
            rmtree(train_save_path + '/' + folders[iteration])
    iteration += 1


folders_with_name = os.listdir(train_save_path)
for i, folder in enumerate(folders_with_name):
    folder_path = train_save_path + '/' + folder
    files = os.listdir(folder_path)
    # print(files)
    for cam, file in enumerate(files):
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, str(i).zfill(3)+ 'c'+str(cam))+'.png')
    os.rename(folder_path, os.path.join(train_save_path, str(i).zfill(3)))
# # ---------------------------------------
# gallery
gallery_save_path = download_path + '/pytorch/gallery'
gallery_size = int(len(folders_with_name)/2)

if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

    image_folders = os.listdir(train_save_path)
    gallery_folder_names = random.sample(image_folders, gallery_size)

    for gallery_folder in gallery_folder_names:
        copy_tree(train_save_path + '/' + gallery_folder, gallery_save_path + '/' + gallery_folder)
        rmtree(train_save_path + '/' + gallery_folder)

# ---------------------------------------
# train_val
val_save_path = download_path + '/pytorch/val'
if not os.path.isdir(val_save_path):
    os.mkdir(val_save_path)

folder_index = -1
folders = ()
for root, dirs, files in os.walk(train_save_path, topdown=True):
    if folder_index == -1:
        folders = dirs
    else:
        name = files[-1]
        dest_folder_name = folders[folder_index]
        src_path = train_save_path + '/' + dest_folder_name + '/' + name
        dst_path = val_save_path + '/' + dest_folder_name
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
        os.remove(src_path)
    folder_index += 1

# -----------------------------------------
# query
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

folder_index = -1
folders = ()
for root, dirs, files in os.walk(gallery_save_path, topdown=True):
    if folder_index == -1:
        folders = dirs
    else:
        name = files[-1]
        dest_folder_name = folders[folder_index]
        src_path = gallery_save_path + '/' + dest_folder_name + '/' + name
        dst_path = query_save_path + '/' + dest_folder_name
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
        os.remove(src_path)
    folder_index += 1