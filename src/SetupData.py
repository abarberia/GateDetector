from cv2 import cv2
import numpy as np
import pickle
import wget
import csv
import os
import re


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    master = os.path.dirname(cwd)

    # Define folder name with the images, masks and csv files for data handler
    folder_imgs = 'WashingtonOBRace'
    csv_name = 'corners.csv'
    img_prefix = 'img'
    mask_prefix = 'mask'

    # Define train and validation txt folder and files
    folder_yolo = 'darknet'
    train_txt = 'train.txt'
    validation_txt = 'validation.txt'
    test_txt = 'test.txt'
    predictions_txt = 'predictions.txt'

    # Define YOLO parameters
    folder_yolo_labels = 'labels'
    folder_yolo_pred = 'predictions_'
    folder_yolo_data = 'data'
    yolo_names = 'yolo.names'
    gate_pairs_pickle = 'gate_pairs.p'
    yolo_video = 'video.avi'
    yolo_default_weights = 'darknet53.conv.74'

    # Define reshape coordinates for the images
    reshape_x = -1
    reshape_y = -1

    # Get image size, folder and speed of video from user input
    reshape_x, reshape_y, folder_yolo_pred, video_speed = \
        user_interface(reshape_x, reshape_y, folder_yolo_pred)

    # Define extra margin for boundary detection [pixels] (+= img width/height)
    extra_x = 1
    extra_y = 1

    # Define train and validation data split, with seed to replicate results
    percentage_train = 80
    percentage_validation = 13
    percentage_test = 7
    np.random.seed(1)

    # Configure names and paths
    folder_imgs = os.path.join(master, folder_imgs)
    folder_yolo = os.path.join(master, folder_yolo)
    train_txt = os.path.join(folder_yolo, train_txt)
    validation_txt = os.path.join(folder_yolo, validation_txt)
    test_txt = os.path.join(folder_yolo, test_txt)
    folder_yolo_labels = os.path.join(master, folder_yolo_labels)
    folder_yolo_pred = os.path.join(master, folder_yolo_pred)
    folder_yolo_data = os.path.join(master, folder_yolo_data)
    predictions_txt = os.path.join(master, folder_yolo, predictions_txt)
    gate_pairs_pickle = os.path.join(folder_yolo_data, gate_pairs_pickle)
    yolo_names = os.path.join(folder_yolo_data, yolo_names)
    yolo_video = os.path.join(folder_yolo_data, yolo_video)

    datahandle = DataHandle(folder_imgs=folder_imgs, csv_name=csv_name,
                            img_prefix=img_prefix, mask_prefix=mask_prefix,
                            shape_x=reshape_x, shape_y=reshape_y)

    # Load data set with training data
    print('\nLoading image dataset from "{:s}"...'.format(folder_imgs))
    datahandle.load_image_dataset()

    yolo_datahandle = YOLO_DataHandle(labels_folder=folder_yolo_labels,
                                      yolo_folder=folder_yolo,
                                      pred_folder=folder_yolo_pred,
                                      data_folder=folder_yolo_data,
                                      train_file=train_txt,
                                      validation_file=validation_txt,
                                      test_file=test_txt,
                                      predictions_file=predictions_txt,
                                      gate_pairs_file=gate_pairs_pickle,
                                      names = yolo_names,
                                      video = yolo_video,
                                      weights = yolo_default_weights,
                                      datahandle=datahandle,
                                      extra_x=extra_x, extra_y=extra_y,
                                      video_speed=video_speed)

    print('Generating img files...')
    yolo_datahandle.generate_img_files()
    print('Generating label files...')
    yolo_datahandle.generate_label_files()
    print('Generating bounding boxes...')
    yolo_datahandle.generate_bounding_boxes()
    print('Splitting train/validation data...')
    yolo_datahandle.split_data(train=percentage_train, 
                               validation=percentage_validation,
                               test=percentage_test)
    print('Generating prediction output files...')
    yolo_datahandle.generate_prediction_output_files()
    yolo_datahandle.save_old_gate_match()
    yolo_datahandle.generate_video()
    yolo_datahandle.download_default_weights()
    print('Test implementation')
    yolo_datahandle.test_bbox_generator('img_51.png')

# User interface for image size, folder name and video speed
def user_interface(default_value_x, default_value_y, folder):
    # Handle image size and folder name
    while True:
        txt_in = str(input('Do you want to resize the images? (Y/n): '))
        txt_in = txt_in.lower()

        valid = True
        if txt_in not in ['yes', 'y', 'no', 'n', '']:
            print(bcolors.WARNING+'Input "{:s}" not valid. Should be format '
                 '("Y"/"n")'.format(txt_in))
            valid = False

        if valid and txt_in in ['', 'no', 'n']:
            txt_in_2 = [default_value_x, default_value_y]

        if valid and txt_in in ['yes', 'y']:
            while True:
                print('\nWhat image size do you want? Enter for default')
                txt_in_2 = str(input('Image size (pixel_width, pixel_height):'
                                     ' '))
                valid_2 = True
                raw_input = txt_in_2
                txt_in_2 = txt_in_2.split()
                txt_in_2 = [x.strip('(').strip(',').strip(')') for x in \
                           txt_in_2]

                if txt_in_2 == []:
                    txt_in_2 = [default_value_x, default_value_y]
                else:
                    if len(txt_in_2) != 2:
                        print(bcolors.WARNING+'Input "{:s}" not in format "(int'
                            ', int)."'.format(raw_input))
                        valid_2 = False
                    else:
                        try:
                            txt_in_2[0] = int(txt_in_2[0])
                        except ValueError:
                            valid_2 = False
                            print(bcolors.WARNING+'Input "{:s}" not in format "'
                                '(int, int)."'.format(raw_input))
                        if valid_2:
                            try: 
                                txt_in_2[1] = int(txt_in_2[1])
                            except ValueError:
                                valid_2 = False
                                print(bcolors.WARNING+'Input "{:s}" not in '
                                     'format "(int, int)."'.format(raw_input))
                if valid_2 and txt_in_2[0] < 1 and txt_in_2[0] != \
                                                    default_value_x:
                    print(bcolors.WARNING+'Values "({:d}, {:d})" are not valid'
                         '.'.format(txt_in_2[0], txt_in_2[1]))
                    valid_2 = False
                if valid_2 and txt_in_2[1] < 1 and txt_in_2[1] != \
                                                    default_value_y:
                    print(bcolors.WARNING+'Values "({:d}, {:d})" are not valid'
                         '.'.format(txt_in_2[0], txt_in_2[1]))
                    valid_2 = False
                if valid_2 and txt_in_2[0] == default_value_x:
                    if txt_in_2[1] != default_value_y:
                        print(bcolors.WARNING+'Values "({:d}, {:d})" are not '
                             'valid.'.format(txt_in_2[0], txt_in_2[1]))
                        valid_2 = False
                if valid_2 and txt_in_2[1] == default_value_y:
                    if txt_in_2[0] != default_value_x:
                        print(bcolors.WARNING+'Values "({:d}, {:d})" are not '
                             'valid.'.format(txt_in_2[0], txt_in_2[1]))
                        valid_2 = False
                if not valid_2:
                    print('Try again'+bcolors.ENDC)
                    continue
                if valid_2:
                    break

        if not valid:
            print('Try again\n'+bcolors.ENDC)
            continue

        if valid:
            width, height = [int(x) for x in txt_in_2]
            break
    if width == default_value_x or height == default_value_y:
        width = default_value_x
        height = default_value_y
        folder_name = folder+'360x360'
    else:
        folder_name = folder+str(width)+'x'+str(height)

    # Handle video speed
    while True:
        txt_in = str(input('How fast do you want the video to be? (int) [1-10]:'
                    ' '))
        valid = True
        if txt_in.strip() == '':
            txt_in = '1'
        try:
            txt_in = int(txt_in)
        except ValueError:
            print(bcolors.WARNING+'Input "{:s}" is not an integer.'
                 .format(txt_in))
            valid = False
        if valid and txt_in < 1:
            print(bcolors.WARNING+'Number "{:d}" is not valid. Value has to be '
                 'positive.'.format(txt_in))
            valid = False
        if valid and txt_in > 10:
            print(bcolors.WARNING+'Number "{:d}" is too high, you will not see '
                 'anything. Try [1-10]'.format(txt_in))
            valid = False
        if not valid:
            print('Try again\n'+bcolors.ENDC)
            continue

        if valid:
            video_speed =txt_in
            break
    return width, height, folder_name, video_speed

# Format image name
def format_name(name):
    name = name[name.index('img_'):]
    if name[-4:] == '.png':
        name = name[:-4]
    if name[-4:] == '.jpg':
        name = name[:-4]
    if name[-4:] == '.txt':
        name = name[:-4]
    return name


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DataHandle:
    def __init__(self, folder_imgs, csv_name, img_prefix, mask_prefix, 
                 shape_x=-1, shape_y=-1):
        self.folder = folder_imgs
        self.files = os.listdir(self.folder)

        self.init_csv(csv_name)
        self.init_img(img_prefix, mask_prefix, shape_x, shape_y)

    # Initialise csv data
    def init_csv(self, csv_name):
        self.csv_name = os.path.join(self.folder, csv_name)
        self.csv = []

        with open(self.csv_name, newline='') as csvfile:
            csv_f = csv.reader(csvfile, delimiter=',')
            for row in csv_f:
                self.csv.append(row)
        self.csv = np.array(self.csv)

    # Initialise image data
    def init_img(self, img_prefix, mask_prefix, shape_x, shape_y):
        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix
        self.shape_x = shape_x
        self.shape_y = shape_y

    # Read images and assign to dictionary
    def load_image_dataset(self):
        self.images = {}
        self.img_names = []
        self.n_images = 0
        for file_name in self.files:
            if self.img_prefix in file_name:
                img_num = file_name.replace(self.img_prefix,'')
                img_name = os.path.join(self.folder, file_name)
                mask_name = os.path.join(self.folder, self.mask_prefix+img_num)

                img_array = cv2.imread(img_name)
                mask_array = cv2.imread(mask_name)

                if self.shape_x != -1 and self.shape_y != -1:
                    # Calculate ratio with original image to scale csv
                    r_x = self.shape_x/img_array.shape[1]
                    r_y = self.shape_y/img_array.shape[0]
                    img_array = cv2.resize(img_array, 
                                          (self.shape_x, self.shape_y))
                    mask_array = cv2.resize(mask_array,
                                           (self.shape_x, self.shape_y))
                else:
                    r_x = 1
                    r_y = 1
                coords = self.mine_gate_coordinates(file_name, r_x, r_y)
                self.images[file_name[:-4]] = [img_array, mask_array, coords]
                self.img_names.append(file_name[:-4])
                self.n_images += 1

    # Get gate coordinates from csv file and scale them
    def mine_gate_coordinates(self, name, ratio_x, ratio_y):
        name = format_name(name)+'.png'
        idxs = np.where(self.csv[:,0] == name)
        coords = self.csv[idxs][:,1:].astype(int)
        for idx, coord in enumerate(coords):
            coords[idx][0::2] = coord[0::2]*ratio_x
            coords[idx][1::2] = coord[1::2]*ratio_y
        return coords

    # Get gate coordinates by image name
    def get_gate_coordinates(self, name):
        name = format_name(name)
        return self.images[name][2]

    # Show image for visualisation
    def show_image(self, name, edited=0, mask=0):
        # Format image name
        name = format_name(name)

        if edited:
            img = self.images[name]
            cv2.imshow('Image', img[0])
            if mask:
                cv2.imshow('Mask', img[1])
        else:
            img_name = os.path.join(self.folder, name+'.png')
            img = self.images[name][0]
            cv2.imshow('Image', img)
            if mask:
                mask_name = os.path.join(self.folder, 'mask'+name[3:]+'.png')
                mask = self.images[name][1]
                cv2.imshow('Mask', mask)
        cv2.waitKey(0)

    # Paint corners for visualisation
    def paint_corners(self, name, img):
        kernel = np.arange(-5, 5)
        black = [0, 0, 0]
        coords = self.get_gate_coordinates(name)

        for coord in coords:
            for pair in range(4):
                x = coord[2*pair]
                y = coord[2*pair+1]
                for i in kernel:
                    for j in kernel:
                        img[y+i][x+j] = black


class YOLO_DataHandle:
    def __init__(self, labels_folder, yolo_folder, pred_folder, data_folder,
                 train_file, validation_file, test_file, predictions_file,
                 gate_pairs_file, names, video, weights, datahandle, extra_x,
                 extra_y, video_speed):
        self.labels_folder = labels_folder
        self.yolo_folder = yolo_folder
        self.pred_folder = pred_folder
        self.data_folder = data_folder
        self.yolo_names = names
        self.yolo_video = video
        self.yolo_default_weights = weights
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.predictions_file = predictions_file
        self.gate_pairs_file = gate_pairs_file
        self.mkdirs()

        self.categories =  [line.rstrip('\n') for line in open(self.yolo_names)]

        self.datahandle = datahandle

        self.extra_x = extra_x
        self.extra_y = extra_y
        self.video_speed = round(video_speed)

    # Make sure required folders are present
    def mkdirs(self):
        try:
            os.mkdir(self.pred_folder)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.labels_folder)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.labels_folder)
        except FileExistsError:
            pass

    # Generate image files in ground truth folder
    def generate_img_files(self):
        for img_name in self.datahandle.img_names:
            img_name = format_name(img_name)+'.png'
            full_img_name = os.path.join(self.labels_folder, img_name)
            cv2.imwrite(full_img_name, self.datahandle.images[img_name[:-4]][0])

    # Generate image txt files in ground truth folder
    def generate_label_files(self):
        self.labels = []
        for img_name in self.datahandle.img_names:
            img_name = format_name(img_name)
            full_img_name = os.path.join(self.labels_folder, img_name+'.txt')
            self.labels.append(img_name+'.txt')
            try:
                open(full_img_name, 'x')
            except FileExistsError:
                continue

    # Generate bounding boxes from csv coordinates
    def generate_bounding_boxes(self):
        self.new_old_gate_match = {}
        for label in self.datahandle.img_names:
            label_png = label+'.png'
            label_txt = label+'.txt'
            img = self.datahandle.images[label][0]
            width = len(img[0])
            height = len(img)

            x_scale = 1./width
            y_scale = 1./height

            label_file_name = os.path.join(self.labels_folder, label_txt)
            label_file = open(label_file_name, "w")

            coords = self.datahandle.get_gate_coordinates(label_png)
            count = 0
            new_old_gate_match = {}
            for coord in coords:
                count += 1
                x_c, y_c, x_w, y_w = self.get_gate_dimensions(coord)
                x_w, y_w = self.bound_edges(x_c, y_c, x_w, y_w, width, height)

                x_center = x_c * x_scale
                y_center = y_c * y_scale
                x_width  = x_w * x_scale
                y_width  = y_w * y_scale

                if count == len(coords):
                    label_file.write("0 " + str(x_center) + " " + str(y_center)\
                         + " " + str(x_width) + " " + str(y_width))
                else:
                    label_file.write("0 " + str(x_center) + " " + str(y_center)\
                        + " " + str(x_width) + " " + str(y_width) + "\n")
                new_old_gate_match[tuple([x_center, y_center, x_width, \
                                          y_width])] = coord
            self.new_old_gate_match[label] = new_old_gate_match

    # Get bounding box dimension from csv coordinates
    def get_gate_dimensions(self, coords):
        coords = np.array(coords)
        x_c = np.average(coords[0::2])
        y_c = np.average(coords[1::2])
        x_w = ((coords[2]+coords[4]) - (coords[0]+coords[6]))/2
        y_w = ((coords[5]+coords[7]) - (coords[1]+coords[3]))/2
        return x_c, y_c, x_w, y_w

    # Limit bounding box edge expansion to image size
    def bound_edges(self, x_c, y_c, x_w, y_w, width, height):
        bound_top = y_c + y_w/2
        bound_bot = y_c - y_w/2
        bound_rig = x_c + x_w/2
        bound_lef = x_c - x_w/2

        margin_x = min(bound_lef, (width-bound_rig), x_w)
        margin_y = min(bound_bot, (height-bound_top), y_w)

        margin_x = margin_x*self.extra_x
        margin_y = margin_y*self.extra_y
        return abs(x_w+margin_x), abs(y_w+margin_y)

    # Split data in training, validation and testing
    def split_data(self, train, validation, test):
        self.percentage_train = train
        self.percentage_validation = validation
        self.percentage_test = test

        # Create and/or truncate train.txt and test.txt
        file_train = open(self.train_file, 'w')
        file_validation = open(self.validation_file, 'w')
        file_test = open(self.test_file, 'w')

        idx_train = round(self.datahandle.n_images * \
                    self.percentage_train / 100)
        idx_validation = round(self.datahandle.n_images * \
                         self.percentage_validation / 100) + idx_train
        idx_test = round(self.datahandle.n_images * \
                         self.percentage_test / 100) + idx_validation

        # Get list of image names and randomise
        img_names = list(self.datahandle.images.keys())
        np.random.shuffle(img_names)

        # Split data between training and validation
        count = 0
        for img_name in img_names:
            full_img_name = os.path.join(self.labels_folder, img_name+'.png\n')
            full_test_name = os.path.join(self.pred_folder, img_name+'.png\n')
            if count < idx_train:
                if count == idx_train-1:
                    full_img_name = full_img_name[:-1]
                file_train.write(full_img_name)
            elif count < idx_validation:
                if count == idx_validation-1:
                    full_img_name = full_img_name[:-1]
                file_validation.write(full_img_name)
            else:
                if count == len(img_names)-1:
                    full_img_name = full_img_name[:-1]
                file_test.write(full_img_name)

            count += 1
        file_train.close()
        file_validation.close()

    # Generate txt file to be used in the predictions
    def generate_prediction_output_files(self):
        with open(self.predictions_file, 'w') as pred_file:
            with open(self.test_file, 'r') as test_file:
                images = test_file.readlines()
                for image in images:
                    name_png = image.strip('\n')
                    img_name = name_png.replace(os.path.dirname(name_png),'')
                    output_name = os.path.join(self.pred_folder, img_name[1:])
                    name = output_name[:-4]
                    line = name_png+' '+name+'\n'
                    pred_file.write(line)

    # Save image gates data to pickle to be accessed by data analyser
    def save_old_gate_match(self):
        pickle.dump(self.new_old_gate_match, open(self.gate_pairs_file, 'wb'))

    # Generate video by concatenating images
    def generate_video(self):
        img_array = []
        sorted_imgs = self.datahandle.img_names
        sorted_imgs.sort(key=self.natural_keys)
        count = 0
        speed_int = int(self.video_speed)
        speed_float = self.video_speed - int(self.video_speed)
        for img_name in sorted_imgs:
            if count%self.video_speed == 0:
                full_img_name = os.path.join(self.labels_folder, img_name+'.png')
                img = self.datahandle.images[img_name][0]
                h, w, l = img.shape
                size = (h, w)
                img_array.append(img)
            count += 1
        out = cv2.VideoWriter(self.yolo_video, cv2.VideoWriter_fourcc(*'DIVX'), 15,\
                              size)
        for img in img_array:
            out.write(img)
        out.release()

    # Function required for image sorting for video
    def atoi(self, text):
        return int(text) if text.isdigit() else text

    # Function required for image sorting for video
    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text) ]

    # Ensure default weights are downloaded from internet for own training
    def download_default_weights(self):
        yolo_files = os.listdir(self.yolo_folder)
        if self.yolo_default_weights not in yolo_files:
            print(bcolors.WARNING+'Default weights used for training YOLO are missing')
            print('Downloading them from "https://pjreddie.com/media/files/darknet53.conv.74"'+bcolors.ENDC)
            url = 'https://pjreddie.com/media/files/darknet53.conv.74'
            path = os.path.join(self.yolo_folder, self.yolo_default_weights)
            wget.download(url, path)

    # Test bounding box generator by showing image
    def test_bbox_generator(self, img):
        img_name = format_name(img)
        full_img_name = os.path.join(self.labels_folder, img_name)
        img = self.datahandle.images[img_name][0]
        height, width = img.shape[:2]

        with open(full_img_name+'.txt', 'r') as file:
            coords = file.readlines()
        thick = 2
        color = (0, 0, 255)
        green = (0, 255, 0)
        for coord in coords:
            cds = np.array(coord.strip('\n').split(' ')).astype(float)[1:]
            x_tl = int((cds[0] - cds[2]/2)*width)
            y_tl = int((cds[1] - cds[3]/2)*height)
            x_br = int((cds[0] + cds[2]/2)*width)
            y_br = int((cds[1] + cds[3]/2)*height)
            start = (x_tl, y_tl)
            end = (x_br, y_br)
            img = cv2.rectangle(img, start, end, color, thick)

        coords_img = self.datahandle.images[img_name][2]
        for coords in coords_img:
        
            img_gate_old = np.array([(coords[0], coords[1]), \
                            (coords[2], coords[3]), \
                            (coords[4], coords[5]), \
                            (coords[6], coords[7]), \
                            (coords[0], coords[1])])
            cv2.polylines(img, [img_gate_old], False, green, 2)

        cv2.imshow('image', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()