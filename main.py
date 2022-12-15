import os
from shutil import rmtree
from PIL import Image
import cv2


# Helper function to save image as grayscale
def save_grayscale(source, destination):
    # Reading an image
    image = cv2.imread(source, cv2.IMREAD_COLOR)
    # Convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Saving an image
    val = cv2.imwrite(destination, gray)

    # img = Image.open(source).convert('L')
    # img.save(destination)


# Checks the boat class from the labels and return the class as a string
def boat_class(txt_file_path):
    file = open(txt_file_path, 'r')
    contents = file.read()
    boat_cls = contents.split(' ')
    return boat_cls[0]


def clean_ID(name):
    id = name.split('_')[0]
    return id


# You only need to change this line to your dataset download path

download_path = 'E:/archive/Ships dataset'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/clean'
# ---------------------------------------
# train
# This takes all the images from train and val and creates train folder
train_path = download_path + '/train/images'
train_label_path = download_path + '/train/labels'
val_path = download_path + '/val/images'
val_label_path = download_path + '/val/labels'
test_path = download_path + '/test/images'
test_label_path = download_path + '/test/labels'

if not os.path.isdir(save_path):
    os.mkdir(save_path)
    for paths in [(train_path, train_label_path), (val_path, val_label_path), (test_path, test_label_path)]:
        for root, dirs, files in os.walk(paths[0], topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                ID = clean_ID(name)
                src_path = paths[0] + '/' + name
                src_label_path = paths[1] + '/' + name[:-3] + 'txt'
                dst_path = save_path + '/' + ID + '_' + boat_class(src_label_path)
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                save_grayscale(src_path, dst_path + '/' + name[:-3] + 'jpg')

# making a list of ship folders with many ships
iteration = -1
folders_with_many_pics = []
folders = ()
for root, dirs, files in os.walk(save_path, topdown=True):
    if iteration == -1:
        folders = dirs
    else:
        if len(files) > 3:
            folders_with_many_pics.append(root.split("\\")[-1])
            rmtree(save_path + '/' + folders[iteration])  # removes folders with more than 3 pics
    iteration += 1

with open('folder_names.txt', 'w') as f:
    for i, line in enumerate(folders_with_many_pics):
        if i < len(folders_with_many_pics) - 1:
            f.write(line + "\n")
        else:
            f.write(line)
