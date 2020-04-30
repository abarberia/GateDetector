from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from cv2 import cv2
import numpy as np
import pickle
import os

from SetupData import format_name, bcolors

def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    master = os.path.dirname(cwd)

    # Define folder name with the images, csv files and predictions
    folder_images = 'labels'
    folders_preds = ['predictions_200x200', 'predictions_650x650', \
                     'predictions_360x360']
    folder_data = 'data'
    folder_src = 'src'
    img_prefix = 'img'
    gate_pairs_pickle = 'gate_pairs.p'

    folders_preds = user_interface(folders_preds, master, folder_images,
                                   folder_data, folder_src, gate_pairs_pickle)

    # Configure names and paths
    folder_images = os.path.join(master, folder_images)
    gate_pairs_pkl = os.path.join(master, folder_data, gate_pairs_pickle)

    # Rescaling factor for gate size to compensate from overprediction
    gate_area_rescale_x = 2
    gate_area_rescale_y = 2
    original_img_width = 360
    original_img_height = 360

    # Set ROC parameters
    range_roc = list(np.linspace(0.05, 1, 20))
    thresh_iou = 0.7

    fprs = []
    tprs = []
    labels = []

    for idx, folder_preds in enumerate(folders_preds):
        print('Analysisng "{:s}" ({:d}/{:d})'.format(folder_preds, idx+1, \
                                              len(folders_preds)))

        folder_preds = os.path.join(master, folder_preds)
        roc = ROC(folder_imgs=folder_images, folder_preds=folder_preds,
                  gate_pairs_file=gate_pairs_pkl,
                  gate_area_rescale_x=gate_area_rescale_x,
                  gate_area_rescale_y=gate_area_rescale_y,
                  rx=original_img_width,
                  ry=original_img_height)

        roc.calc_roc(range_roc=range_roc, thresh_iou=thresh_iou)
        fpr, tpr, labels = roc.plot_roc()

        fprs.append(fpr)
        tprs.append(tpr)
        labels.append(labels)

    for coordinates in zip(fprs, tprs, folders_preds):
        plt.plot(coordinates[0], coordinates[1], label=coordinates[2])
    plt.plot([0,1], [0,1])

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('IOU={:.2f}'.format(thresh_iou), fontsize=14)
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.grid()
    # plt.legend(fontsize=14, bbox_to_anchor=(1, 0.5))
    plt.legend(fontsize=14)
    plt.show()

    # Test implementation
    folder_preds = os.path.join(master, 'predictions_360x360')
    roc = ROC(folder_imgs=folder_images, folder_preds=folder_preds,
                gate_pairs_file=gate_pairs_pkl,
                gate_area_rescale_x=gate_area_rescale_x,
                gate_area_rescale_y=gate_area_rescale_y, 
                rx=original_img_width,
                ry=original_img_height)
    roc.calc_roc(range_roc=[0.6], thresh_iou=thresh_iou)
    roc.test_threshold('img_270.jpg')


# User interface for folder name
def user_interface(default_values, master, folder_images, folder_data, 
                   folder_src, data_pckl):
    my_dirs = []
    for d in os.listdir(master):
        if os.path.isdir(os.path.join(master, d)):
            my_dirs.append(d)

    parent_folder = os.path.join(master, folder_data)
    if folder_images not in os.listdir(master):
        cwd = os.getcwd()
        folder_to_run = os.path.join(master.replace(cwd,''),folder_src,'a')[:-1]
        print(bcolors.WARNING+'\nGround truth folder "{:s}" not there. Make '
              'sure to run "python3 {:s}SetupData.py" before running this '
              'file\n'.format(folder_images, folder_to_run))
        quit()

    if data_pckl not in os.listdir(parent_folder):
        cwd = os.getcwd()
        folder_to_run = os.path.join(master.replace(cwd,''),folder_src,'a')[:-1]
        print(bcolors.WARNING+'\nGround truth pickle "{:s}" not in "{:s}".'
              'Make sure to run "python3 {:s}SetupData.py" before running this '
              'file\n'.format(data_pckl, parent_folder, folder_to_run))
        quit()


    while True:
        txt_in = str(input('Write name of folder(s) with predicted images '
                           '(Enter for default, "1" to see options.): '))

        if txt_in == '':
            for value in default_values:
                txt_in+=value+' '

        folders_in = txt_in.strip().split(' ')

        valid = True
        if folders_in == ['1']:
            print('Options are:')
            for folder in my_dirs:
                if folder[:11] == 'predictions':
                    print('\t', folder)
            valid = False
            continue

        if valid:
            for folder_in in folders_in:
                if valid and folder_in not in my_dirs:
                    print(bcolors.WARNING + '\nCould not find folder "{:s}"'\
                        .format(folder_in))
                    valid = False
                if valid and len(folder_in) < len('predictions'):
                    print(bcolors.WARNING + '\nInvalid folder selection "{:s}"'
                        '.'.format(folder_in))
                    valid = False
                if valid and folder_in[:11] != 'predictions':
                    print(bcolors.WARNING + '\nInvalid folder selection "{:s}"'
                        '.'.format(folder_in))
                    valid = False
            if not valid:
                print('Options are:')
                for folder in my_dirs:
                    if folder[:11] == 'predictions':
                        print('\t', folder)
                print()

        if valid:
            for folder_in in folders_in:
                files = os.listdir(os.path.join(master, folder_in))
                files_labels = os.listdir(os.path.join(master,folder_images))
                if len(files) == 0:
                    print(bcolors.WARNING+'Folder "{:s}" is empty.'\
                        .format(folder_in))
                    valid = False
                for file in files:
                    if file[-4:] == '.txt':
                        if file not in files_labels:
                            print(bcolors.WARNING + '\nCould not find "{:s}"'
                                'on folder "{:s}" in ground truth folder "{:s}"'
                                ''.format(file, folder_in, folder_images))
                            print('Remove the file "{:s}" or add its ground '
                                'truth to the folder "{:s}"'.format(file, \
                                folder_images))
                            valid = False
                            quit()
        if not valid:
            print('Try again\n'+bcolors.ENDC)
            continue

        if valid:
            break
    return folders_in


class ROC:
    def __init__(self, folder_imgs, folder_preds, gate_pairs_file, 
                 gate_area_rescale_x, gate_area_rescale_y, rx, ry):
        self.folder_imgs = folder_imgs
        self.folder_preds = folder_preds
        self.gate_pairs_file = gate_pairs_file
        self.gate_area_rescale_x = gate_area_rescale_x
        self.gate_area_rescale_y = gate_area_rescale_y
        self.rx = float(rx)
        self.ry = float(ry)

        self.init_preds()
        self.init_imgs()

    # Initialise prediction data
    def init_preds(self):
        self.img_names = []
        self.predicted_gates = {}
        for file in os.listdir(self.folder_preds):
            if file[-4:] == '.txt':
                img_name = format_name(file)
                full_name_txt = os.path.join(self.folder_preds, file)

                self.img_names.append(img_name)

                gate_list = []
                with open(full_name_txt, 'r') as txt_file:
                    gates = txt_file.readlines()
                    for gate in gates:
                        coord_list = []
                        coords =  gate.strip(' \n').split(' ')
                        for coord in coords:
                            coord_list.append(float(coord))
                        coords = np.array(coord_list[:-1])
                        prob = coord_list[-1]
                        gate_coord = [coords, prob]
                        gate_list.append(np.array(gate_coord))
                    self.predicted_gates[img_name] = gate_list

    # Initialise ground truth dta
    def init_imgs(self):
        self.img_gates = {}
        for img_name in self.img_names:
            full_name_txt = os.path.join(self.folder_imgs, img_name+'.txt')
            gate_list = []
            with open(full_name_txt, 'r') as txt_file:
                gates = txt_file.readlines()
                for gate in gates:
                    coord_list = []
                    coords = gate.strip('\n').split(' ')[1:]
                    for coord in coords:
                        coord_list.append(float(coord))
                    gate_list.append(coord_list)
                self.img_gates[img_name] = gate_list
        self.new_old_gate_match = pickle.load(open(self.gate_pairs_file, 'rb'))

    # Calculate ROC curve
    def calc_roc(self, range_roc, thresh_iou):
        self.tpr = []
        self.fpr = []
        self.fpavg = []
        self.labels = []
        for thresh in range_roc:
            self.tp = 0
            self.fp = 0
            self.fn = 0
            self.tn = 0
            self.thresh = thresh

            self.apply_threshold(thresh)
            self.associate_gates()
            self.calc_confusion_matrix(thresh_iou)
            # print("({:f})->TP, FP, FN, TN: {:d}, {:d}, {:d}, {:f}"\
            # .format(thresh, self.tp, self.fp, self.fn, self.tn))

            if self.tp + self.fn == 0:
                tpr = 0
            else:
                tpr = self.tp / (self.tp + self.fn)
            if self.fp + self.tn == 0:
                fpr = 0
            else:
                fpr = self.fp / (self.fp + self.tn)
            fpavg = self.fp / len(self.img_names)
            if self.tp + self.fp == 0:
                fpavg = 0
            else:
                fpavg = self.fp / (self.tp + self.fp)
            self.tpr.append(tpr)
            self.fpr.append(fpr)
            self.fpavg.append(fpavg)
            self.labels.append('{:.2f}'.format(thresh))

    # Apply threshold to what is considered a gate
    def apply_threshold(self, thresh):
        self.pred_gates = {}
        for img_name in self.img_names:
            updated_list = []
            for gate in self.predicted_gates[img_name]:
                if gate[1] > thresh:
                    updated_list.append(gate)
            self.pred_gates[img_name] = np.array(updated_list)

    # Associate ground truth gates and predicted gates
    def associate_gates(self):
        self.gate_pairs = {}
    
        for img_name in self.img_names:
            if len(self.img_gates[img_name]) > 0:
                gates_img  = self.img_gates[img_name]
            else:
                gates_img = []
            if len(self.pred_gates[img_name]) > 0:
                gates_pred = self.pred_gates[img_name][:,0]
            else:
                gates_pred = []
            if len(gates_img) > len(gates_pred):
                if len(gates_pred) == 0:
                    gate_pairs = []
                    for i, _ in enumerate(gates_img):
                        gate_pairs.append([i, None])
                    self.gate_pairs[img_name] = gate_pairs

                else:
                    gate_pairs = []
                    for i, _ in enumerate(gates_img):
                        gate_pair = []
                        coords_img = np.array(gates_img[i])
                        min_dist = 1e9
                        closest_gate = None

                        for j, _ in enumerate(gates_pred):
                            coords_pred = np.array(gates_pred[j])
                            dist = np.linalg.norm(coords_img[:2] - \
                                                  coords_pred[:2])
                            size_diff = abs(coords_img[2] - coords_pred[2]) + \
                                        abs(coords_img[3] - coords_pred[3])
                        
                            if (coords_pred[2] * coords_pred[3]) < 1e-4:
                                size_diff = 1

                            gate_pair.append([i, j, dist*size_diff])
                        gate_pairs.append(gate_pair)

                    gate_pairs = np.array(gate_pairs)
                    actual_pairs = []
                    tmp_shape = np.shape(gate_pairs)

                    for j, _ in enumerate(gates_pred):
                        closest_pair, gate_pairs = \
                            self.closest_in_list(gate_pairs)
                        actual_pairs.append(closest_pair)
                    actual_pairs = np.array(actual_pairs)
                    for i in range(tmp_shape[0]):
                        if i not in actual_pairs[:,0]:
                            actual_pairs = np.append(actual_pairs, \
                                                     [[i, None]], axis=0)
                    self.gate_pairs[img_name] = actual_pairs

            elif len(gates_img) < len(gates_pred):
                if len(gates_img) == 0:
                    gate_pairs = []
                    for i, _ in enumerate(gates_pred):
                        gate_pairs.append([None, i])
                    self.gate_pairs[img_name] = gate_pairs

                else:
                    gate_pairs = []
                    for i, _ in enumerate(gates_pred):
                        gate_pair = []
                        coords_pred = np.array(gates_pred[i])
                        min_dist = 1e9
                        closest_gate = None

                        for j, _ in enumerate(gates_img):
                            coords_img = np.array(gates_img[j])
                            dist = np.linalg.norm(coords_img[:2] - \
                                                  coords_pred[:2])
                            size_diff = abs(coords_img[2] - coords_pred[2]) + \
                                        abs(coords_img[3] - coords_pred[3])

                            if (coords_pred[2] * coords_pred[3]) < 2e-4:
                                size_diff = 1
                            gate_pair.append([j, i, dist*size_diff])
                        gate_pairs.append(gate_pair)
                    gate_pairs = np.array(gate_pairs)
                    actual_pairs = []
                    tmp_shape = np.shape(gate_pairs)
                    for j, _ in enumerate(gates_img):
                        closest_pair, gate_pairs = \
                            self.closest_in_list(gate_pairs)
                        actual_pairs.append(closest_pair)
                    actual_pairs = np.array(actual_pairs)
                    for i in range(tmp_shape[0]):
                        if i not in actual_pairs[:,1]:
                            actual_pairs = np.append(actual_pairs, [[None, i]],\
                                                     axis=0)
                    self.gate_pairs[img_name] = actual_pairs

            elif len(gates_img) == len(gates_pred):
                if len(gates_img) == 0:
                    self.gate_pairs[img_name] = [-1, -1]

                else:
                    gate_pairs = []
                    
                    for i, _ in enumerate(gates_pred):
                        gate_pair = []
                        coords_pred = np.array(gates_pred[i])
                        min_dist = 1e9
                        closest_gate = None

                        for j, _ in enumerate(gates_img):
                            coords_img = np.array(gates_img[j])
                            dist = np.linalg.norm(coords_img[:2] - \
                                                  coords_pred[:2])
                            size_diff = abs(coords_img[2] - coords_pred[2]) + \
                                        abs(coords_img[3] - coords_pred[3])

                            if (coords_pred[2] * coords_pred[3]) < 1e-4:
                                size_diff = 1
                            gate_pair.append([j, i, dist*size_diff])
                        gate_pairs.append(gate_pair)

                    gate_pairs = np.array(gate_pairs)
                    actual_pairs = []
                    tmp_shape = np.shape(gate_pairs)

                    for j in range(tmp_shape[0]):
                        closest_pair, gate_pairs = \
                            self.closest_in_list(gate_pairs)
                        actual_pairs.append(closest_pair)
                    actual_pairs = np.array(actual_pairs)
                    self.gate_pairs[img_name] = actual_pairs

    # Find gate closest to ground truth
    def closest_in_list(self, gate_pairs):
        tmp_shape = np.shape(gate_pairs)[:2]
        dists = gate_pairs[:,:,2]
        min_idx = np.argmin(dists)
        tmp_lst = np.ravel(dists)

        min_row = min_idx//tmp_shape[1]
        min_col = min_idx%tmp_shape[1]
        pair = gate_pairs[min_row, min_col][:2]

        new_pairs = np.concatenate((gate_pairs[:min_row], \
                                    gate_pairs[min_row+1:]), axis=0)
        new_pairs = np.concatenate((new_pairs[:,:min_col], \
                                    new_pairs[:,min_col+1:]), axis=1)
        return pair, new_pairs

    # Calculate confusion matrix
    def calc_confusion_matrix(self, thresh_iou):
        for img_name in self.img_names:
            gate_area = 0
            union_area = 0
            n_gates = 0
            for gate in self.gate_pairs[img_name]:
                # Gate pairs are: (idx img, idx pred)
                if gate[0] == None:
                    self.fp += 1
                    gate_area_i = self.calc_gate_area_i(img_name, gate)
                    union_area_i = gate_area_i
                    n_gates += 1
                elif gate[1] == None:
                    self.fn += 1
                    gate_area_i = 0
                    union_area_i = self.calc_polygon_area_i(img_name, gate)
                else:
                    iou, gate_area_i, union_area_i = self.calc_iou(img_name, 
                                                                   gate)
                    if iou > thresh_iou:
                        self.tp += 1
                    else:
                        self.fn += 1
                        self.fp += 1
                    n_gates += 1
                gate_area += gate_area_i
                union_area += union_area_i
            if n_gates > 0:
                avg_gate_area = gate_area/n_gates
                free_area = 1 - union_area
                self.tn += free_area / avg_gate_area

    # Calculate area of gate
    def calc_gate_area_i(self, img_name, gate):
        coord_pred = self.pred_gates[img_name][int(gate[1]), 0]
        pred_lef = coord_pred[0]-coord_pred[2]/2/self.gate_area_rescale_x
        pred_rig = coord_pred[0]+coord_pred[2]/2/self.gate_area_rescale_x
        pred_bot = coord_pred[1]-coord_pred[3]/2/self.gate_area_rescale_y
        pred_top = coord_pred[1]+coord_pred[3]/2/self.gate_area_rescale_y
        pred_area = (pred_rig - pred_lef) * (pred_top - pred_bot)
        return pred_area

    def calc_polygon_area_i(self, img_name, gate):
        coord_img = self.img_gates[img_name][int(gate[0])]
        coord_img_old = self.new_old_gate_match[img_name][tuple(coord_img)]

        coord_img_old = np.array(coord_img_old).astype(float)
        coord_img_old[0::2] = coord_img_old[0::2]/self.rx
        coord_img_old[1::2] = coord_img_old[1::2]/self.ry

        img_gate_old = Polygon([(coord_img_old[0], coord_img_old[1]), \
                                (coord_img_old[2], coord_img_old[3]), \
                                (coord_img_old[4], coord_img_old[5]), \
                                (coord_img_old[6], coord_img_old[7])])
        return img_gate_old.area

    # Calculate IOU
    def calc_iou(self, img_name, gate):
        # Coords are in format(x_center, y_center, x_width, y_width)
        coord_img = self.img_gates[img_name][int(gate[0])]
        coord_img_old = self.new_old_gate_match[img_name][tuple(coord_img)]
        coord_pred = self.pred_gates[img_name][int(gate[1]), 0]

        coord_img_old = np.array(coord_img_old).astype(float)
        coord_img_old[0::2] = coord_img_old[0::2]/self.rx
        coord_img_old[1::2] = coord_img_old[1::2]/self.ry

        pred_lef = coord_pred[0]-coord_pred[2]/2/self.gate_area_rescale_x
        pred_rig = coord_pred[0]+coord_pred[2]/2/self.gate_area_rescale_x
        pred_bot = coord_pred[1]-coord_pred[3]/2/self.gate_area_rescale_y
        pred_top = coord_pred[1]+coord_pred[3]/2/self.gate_area_rescale_y

        img_gate_old = Polygon([(coord_img_old[0], coord_img_old[1]), \
                                (coord_img_old[2], coord_img_old[3]), \
                                (coord_img_old[4], coord_img_old[5]), \
                                (coord_img_old[6], coord_img_old[7])])
        pred_gate = Polygon([(pred_lef, pred_top), (pred_rig, pred_top), \
                             (pred_rig, pred_bot), (pred_lef, pred_bot)])

        overlap = img_gate_old.intersection(pred_gate).area
        union = img_gate_old.area + pred_gate.area - overlap

        return overlap/union, img_gate_old.area, union

    # Get FPR, TPS and labels
    def plot_roc(self):
        self.fpr = [1] + self.fpr
        self.tpr = [1] + self.tpr
        self.labels = ['0.0'] + self.labels

        # coords_used = []
        # for label, tpr, fpr in zip(self.labels, self.tpr, self.fpr):
        #     if tuple([fpr, tpr]) not in coords_used:
        #         plt.annotate(label, (fpr, tpr))
        #         coords_used.append(tuple([fpr, tpr]))
    
        return self.fpr, self.tpr, self.labels

    # Test threshold calculator by showing output gates on image
    def test_threshold(self, img_name):
        img_name = format_name(img_name)
        full_img_name = os.path.join(self.folder_imgs, img_name)
        img = cv2.imread(full_img_name+'.png')
        height, width = img.shape[:2]

        with open(full_img_name+'.txt', 'r') as file:
            coords = file.readlines()
        thick = 2
        black = (0, 0, 255)
        green = (0, 255, 0)

        for gate in self.gate_pairs[img_name]:
            if gate[0] != None:
                coord_img = self.img_gates[img_name][int(gate[0])]
                coord_img_old = self.new_old_gate_match[img_name]\
                                                       [tuple(coord_img)]

                img_gate_old = np.array([(coord_img_old[0], coord_img_old[1]), \
                                (coord_img_old[2], coord_img_old[3]), \
                                (coord_img_old[4], coord_img_old[5]), \
                                (coord_img_old[6], coord_img_old[7]), \
                                (coord_img_old[0], coord_img_old[1])])
                cv2.polylines(img, [img_gate_old], False, green, 2)

            if gate[1] != None:
                coord_pred = self.pred_gates[img_name][int(gate[1]), 0]
                pred_lef = int((coord_pred[0]-coord_pred[2]/2/self\
                           .gate_area_rescale_x)*self.rx)
                pred_rig = int((coord_pred[0]+coord_pred[2]/2/self\
                           .gate_area_rescale_x)*self.rx)
                pred_bot = int((coord_pred[1]-coord_pred[3]/2/self\
                           .gate_area_rescale_y)*self.ry)
                pred_top = int((coord_pred[1]+coord_pred[3]/2/self\
                           .gate_area_rescale_y)*self.ry)

                pred_gate = np.array([(pred_lef, pred_top), (pred_rig, \
                            pred_top), (pred_rig, pred_bot), (pred_lef, \
                            pred_bot), (pred_lef, pred_top)])
                cv2.polylines(img, [pred_gate], False, black, 2)

        cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
