import os
import cv2

# ----------------------------------------------------------------------------
def write_labels(vector, label_save_path, filename):
    with open(os.path.join(label_save_path, filename[:-3] + 'txt'), 'w') as f:
        f.write(vector)


if __name__ == '__main__':

    image_path = 'E:/archive/Ships dataset/clean/000'

    first = True
    for root, dirs, files in os.walk(image_path, topdown=True):
        if first:
            first = False
            continue
        else:
            image_path = os.path.join(root, files[0])
            name = root.split('\\')[-1]
            img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
            cv2.imshow(name, img)
            cv2.waitKey(0)
            orientation = input("Enter orientation vector:")
            cv2.destroyAllWindows()
            for image in files:
                write_labels(orientation, root, image)