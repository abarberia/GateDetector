# import keras_preprocessing as kp
# import tensorflow as tf
import matplotlib as mpl
import numpy as np
from cv2 import cv2
import csv
import os

from GateDetector import format_name

def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    master = os.path.dirname(cwd)

    # Define folder name with the images, masks and csv files
    folder_imgs = 'WashingtonOBRace'
    csv_name = 'corners.csv'
    img_prefix = 'img'
    mask_prefix = 'mask'

    # Define YOLO parameters
    yolo_labels_folder = 'labels'
    yolo_cfg_folder = 'cfg'
    yolo_names = 'yolo.names'
    label_names = 'labels.names'

    # Define extra margin for boundary detection [pixels]
    extra_x = 0.6
    extra_y = 0.6

    # Configure names and paths
    folder_imgs = os.path.join(master, folder_imgs)
    yolo_labels_folder = os.path.join(master, yolo_labels_folder)
    yolo_cfg_folder = os.path.join(master, yolo_cfg_folder)

    # Initialise data generator with the folder name
    data_generator = DataGenerator(folder_imgs=folder_imgs, csv_name=csv_name, 
                                img_prefix=img_prefix, mask_prefix=mask_prefix)

    # Get the current images and csv
    data_generator.get_current_images()
    data_generator.get_csv()

    data_generator.get_image_properties()

    # Modify data
    data_generator.modify_images()


class DataGenerator:
    def __init__(self, folder_imgs, csv_name, img_prefix, mask_prefix):
        # Initialise images folder and file names
        self.folder_imgs = folder_imgs

        self.files = os.listdir(self.folder_imgs)
        self.csv_name = csv_name

        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix

        # Initialise keras image processor
        # self.kp = kp.image.image_data_generator()

    def get_current_images(self):
        self.img_names  = []
        self.mask_names = []

        for file_name in self.files:
            if self.img_prefix in file_name:
                self.img_names.append(file_name)
                termination = file_name.replace(self.img_prefix,'')
                self.mask_names.append(self.mask_prefix+termination)

    def get_csv(self):
        self.csv_name = os.path.join(self.folder_imgs, self.csv_name)
        self.csv = []

        with open(self.csv_name, newline='') as csvfile:
            csv_f = csv.reader(csvfile, delimiter=',')
            for row in csv_f:
                self.csv.append(row)
        self.csv = np.array(self.csv)

    def get_image_properties(self):
        img = cv2.imread(os.path.join(self.folder_imgs, self.img_names[0]))
        mask = cv2.imread(os.path.join(self.folder_imgs, self.mask_names[0]))

        print('Image propertires:')
        print('\tImage size: {:d} x {:d} x {:d}'.format(img.shape[0], img.shape[1], img.shape[2]))
        print('Mask properties')
        print('\tMask size: {:d} x {:d} x {:d}'.format(mask.shape[0], mask.shape[1], mask.shape[2]))

    def get_gate_coordinates(self, name):        
        name = format_name(name)+'.png'
        idxs = np.where(self.csv[:,0] == name)
        coords = self.csv[idxs][:,1:]
        return coords.astype(int)

    def modify_images(self):
        return

if __name__ == '__main__':
    main()