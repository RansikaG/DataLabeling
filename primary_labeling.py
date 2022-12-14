import os
import cv2

# ----------------------------------------------------------------------------
def write_labels(orientation, filename):
    with open(os.path.join(label_save_path, filename[:-3] + 'txt'), 'w') as f:
        f.write(orientation)


if __name__ == '__main__':

    image_folder_path = "F:/Downloads/boat dataset/online/5th"
    label_save_path = "F:/Downloads/boat dataset/labels"

    if not os.path.isdir(label_save_path):
        os.mkdir(label_save_path)

    for root, dirs, files in os.walk(image_folder_path, topdown=True):
        for i, name in enumerate(files):
            if not name[-3:] == 'jpg':
                continue
            image_path = os.path.join(image_folder_path, name)
            img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
            cv2.imshow(name, img)
            cv2.waitKey(0)
            orientation = input("orientation class:")
            cv2.destroyAllWindows()  # destroy all windows
            if orientation == 'b' and i != 0:
                previous_name=files[i-1]
                cv2.destroyAllWindows()  # destroy all windows
                image_path = os.path.join(image_folder_path, previous_name)
                img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
                cv2.imshow(previous_name, img)
                cv2.waitKey(0)
                orientation = input("Enter correct orientation class:")
                cv2.destroyAllWindows()  # destroy all windows
                write_labels(orientation, previous_name)
            else:
                write_labels(orientation, name)

