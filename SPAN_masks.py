import csv
import os


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


if __name__ == '__main__':
    csv_path = 'F:/Downloads/data/dataset.csv'
    image_path = 'F:/Downloads/data/2/images'
    txt_file_path = 'F:/Downloads/data/2/labels'
    # createCSV_file_for_2(csv_path, image_path, txt_file_path)

    front_img_path = 'F:/Downloads/data/front_rear_images/front'
    rear_img_path = 'F:/Downloads/data/front_rear_images/rear'
    update_additional_front_rear(csv_path, front_img_path, rear_img_path)

    # https://dms.uom.lk/apps/files/?dir=/&fileid=7629451
