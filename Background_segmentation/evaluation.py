import cv2
import os
import numpy as np
from collections import defaultdict


class Point:
    """
    class to represent a point
    """
    def __init__(self, x, y):
        """
        constructor
        :param x: x coordinate
        :param y: y coordinate
        """
        self.x = x
        self.y = y


class Rectangle:
    """
    class to represent a rectangle
    """
    def __init__(self, x_tl, y_tl, x_br, y_br):
        """
        constructor
        :param x_tl: x coordinate, top left
        :param y_tl: y coordinate, top left
        :param x_br: x coordinate, bottom right
        :param y_br: y coordinate, bottom right
        """
        self.x_tl = x_tl
        self.y_tl = y_tl
        self.x_br = x_br
        self.y_br = y_br
        self.center_point = Point(round(x_tl + (self.width() / 2)), round(y_tl + (self.height() / 2)))

    def width(self):
        """
        computes the width of the rectangle
        :return: width
        """
        return self.x_br - self.x_tl

    def height(self):
        """
        computes the height of the rectangle
        :return: height
        """
        return self.y_br - self.y_tl

    def area(self):
        """
        computes the area of a rectangle
        :return: area
        """
        return self.width() * self.height()

    def intersection_area(self, rect):
        """
        computes the area of the intersection between 2 rectangles
        :param rect: other rectangle to do the computation
        :return: area
        """
        dx = min(self.x_br, rect.x_br) - max(self.x_tl, rect.x_tl)
        dy = min(self.y_br, rect.y_br) - max(self.y_tl, rect.y_tl)
        if dx >= 0 and dy >= 0:
            return dx * dy
        else:
            return 0.0

    @staticmethod
    def intersection_over_union(rect1, rect2):
        intersect_area = rect1.intersection_area(rect2)
        union_area = rect1.area() + rect2.area() - intersect_area
        return intersect_area / union_area


class LobsterReader:
    def __init__(self, file):
        self.file = file
        self.boxes = defaultdict(list)

    def read(self):
        with open(self.file) as f:
            for line in f:
                split_line = line.split(' ')
                frame_id = split_line[0]
                self.boxes[int(frame_id)].append(Rectangle(int(split_line[1]), int(split_line[2]), int(split_line[1]) + int(split_line[3]), int(split_line[2]) + int(split_line[4])))


class RCNNReader:
    def __init__(self, file):
        self.file = file
        self.boxes = defaultdict(list)

    def read(self, flag):
        with open(self.file) as f:
            for line in f:
                split_line = line.split(" ")
                frame_ID = split_line[1]
                split_line = split_line[2:]
                if len(split_line) != 0:
                    for nbr_box in range(0, len(split_line) // 8):
                        if flag == 2:
                            self.boxes[int(frame_ID)].append(Rectangle(int(float(split_line[8 * nbr_box + 1])),
                                                                       int(float(split_line[8 * nbr_box + 5]) + 87),
                                                                       int(float(split_line[8 * nbr_box + 3])),
                                                                       int(float(split_line[8 * nbr_box + 7]) + 87)))
                        else:
                            self.boxes[int(frame_ID)].append(Rectangle(int(float(split_line[8 * nbr_box + 1])),
                                                                       int(float(split_line[8 * nbr_box + 5])),
                                                                       int(float(split_line[8 * nbr_box + 3])),
                                                                       int(float(split_line[8 * nbr_box + 7]))))


class GTReader:
    def __init__(self, filenames):
        self.filenames = filenames
        self.gt = defaultdict(list)

    def extract_boxes(self, ID_offset, mask_name, flag):
        i = ID_offset
        for file in self.filenames:
            img = cv2.imread(file, 0)
            mask = cv2.imread(mask_name, 0)
            if flag == 2 or flag == 3 :
                img = cv2.bitwise_and(img, mask)
            img_show = cv2.imread(file, 1)
            img[img < 200] = 0
            contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > 300 and flag == 0:
                    self.gt[i].append(Rectangle(x, y, x + w, y + h))
                    cv2.rectangle(img_show, (x, y), (x+w, y+h), (0, 0, 255))
                elif y >= 1 and flag == 1:
                    self.gt[i].append(Rectangle(x, y, x + w, y + h))
                    cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 0, 255))
                elif w * h > 200 and flag == 2:
                    self.gt[i].append(Rectangle(x, y, x + w, y + h))
                    cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 0, 255))
                elif w * h > 1000 and flag == 3:
                    self.gt[i].append(Rectangle(x, y, x + w, y + h))
                    cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 0, 255))
            #cv2.imshow('gt', img)
            #cv2.imshow('boxes', img_show)
            #cv2.waitKey(0)
            i += 1


def visualize_boxes(frame, lob_boxes, rcnn_boxes, gt_boxes):
    lob_img = cv2.imread(frame)
    rcnn_img = cv2.imread(frame)
    for gt in gt_boxes:
        cv2.rectangle(lob_img, (gt.x_tl, gt.y_tl), (gt.x_br, gt.y_br), (0, 0, 255), thickness=2)
        cv2.rectangle(rcnn_img, (gt.x_tl, gt.y_tl), (gt.x_br, gt.y_br), (0, 0, 255), thickness=2)
    for box in lob_boxes:
        cv2.rectangle(lob_img, (box.x_tl, box.y_tl), (box.x_br, box.y_br), (255, 0, 0), thickness=2)
    for box in rcnn_boxes:
        cv2.rectangle(rcnn_img, (box.x_tl, box.y_tl), (box.x_br, box.y_br), (0, 255, 0), thickness=2)
    cv2.imshow('lobster', lob_img)
    cv2.imshow('rcnn', rcnn_img)
    cv2.waitKey(0)


def get_statistics(lobster, rcnn, gt, input_files, offset):
    tp_lob = 0
    fp_rcnn = 0
    fn_lob = 0
    tp_rcnn = 0
    fp_lob = 0
    fn_rcnn = 0
    bboverlap = 0.5
    for key in gt.keys():
        # print('Processing {} of {}'.format(key - offset, len(gt)))
        lobster_boxes = lobster[key]
        rcnn_boxes = rcnn[key]
        gt_boxes = gt[key]
        lobster_mat = np.zeros(shape=(len(gt_boxes), len(lobster_boxes)))
        rcnn_mat = np.zeros(shape=(len(gt_boxes), len(rcnn_boxes)))
        for i in range(0, len(gt_boxes)):
            for j in range(0, len(lobster_boxes)):
                lobster_mat[i, j] = Rectangle.intersection_over_union(gt_boxes[i], lobster_boxes[j])
            for j in range(0, len(rcnn_boxes)):
                rcnn_mat[i, j] = Rectangle.intersection_over_union(gt_boxes[i], rcnn_boxes[j])

        lobster_mat[lobster_mat < bboverlap] = 0
        id_lob = []
        for i in range(0, lobster_mat.shape[0]):
            if any(lobster_mat[i, :]):
                id_lob.append(np.argmax(lobster_mat[i, :]))
            else:
                fn_lob += 1
        tp_lob += len(id_lob)
        fp_lob += len(lobster_boxes) - len(id_lob)

        rcnn_mat[rcnn_mat < bboverlap] = 0
        id_rcnn = []
        for i in range(0, rcnn_mat.shape[0]):
            if any(rcnn_mat[i, :]):
                id_rcnn.append(np.argmax(rcnn_mat[i, :]))
            else:
                fn_rcnn += 1
        tp_rcnn += len(id_rcnn)
        fp_rcnn += len(rcnn_boxes) - len(id_rcnn)
        #if offset == 800:
        #    visualize_boxes(input_files[key - offset], lobster_boxes, rcnn_boxes, gt_boxes)

    return tp_lob, fp_lob, fn_lob, tp_rcnn, fp_rcnn, fn_rcnn


def main():
    path = os.path.dirname(os.path.abspath(__file__))
    baseline_lob = os.path.join(path, 'lobster', 'baseline.txt')
    night_lob = os.path.join(path, 'lobster', 'night.txt')
    cam_jitter_lob = os.path.join(path, 'lobster', 'cam_jitter.txt')
    dyn_back_lob = os.path.join(path, 'lobster', 'dyn_back.txt')
    baseline_rcnn = os.path.join(path, 'rcnn', 'PETS2006_results', 'pets_boxes.txt')
    dyn_back_rcnn = os.path.join(path, 'rcnn', 'Fall_results', 'fall_boxes.txt')
    cam_jitter_rcnn = os.path.join(path, 'rcnn', 'sidewalk_results', 'sidewalk_boxes.txt')
    night_rcnn = os.path.join(path, 'rcnn', 'streetCornerAtNight_results', 'streetCornerAtNight_boxes.txt')

    dataset_path = '/home/beaupreda/litiv/datasets/CD2014'
    baseline_gt_path = os.path.join(dataset_path, 'baseline', 'groundtruth')
    night_gt_path = os.path.join(dataset_path, 'night', 'groundtruth')
    dyn_back_gt_path = os.path.join(dataset_path, 'dynamic_background', 'groundtruth')
    cam_jitter_gt_path = os.path.join(dataset_path, 'camera_jittering', 'groundtruth')

    gt_baseline = os.listdir(baseline_gt_path)
    gt_night = os.listdir(night_gt_path)
    gt_cam_jitter = os.listdir(cam_jitter_gt_path)
    gt_dyn_back = os.listdir(dyn_back_gt_path)

    gt_baseline = [os.path.join(baseline_gt_path, f) for f in gt_baseline]
    gt_baseline.sort()
    gt_baseline = gt_baseline[299:1199] # 1200
    gt_night = [os.path.join(night_gt_path, f) for f in gt_night]
    gt_night.sort()
    gt_night = gt_night[799:2999] # 3000
    gt_cam_jitter = [os.path.join(cam_jitter_gt_path, f) for f in gt_cam_jitter]
    gt_cam_jitter.sort()
    gt_cam_jitter = gt_cam_jitter[799:1199]
    gt_dyn_back = [os.path.join(dyn_back_gt_path, f) for f in gt_dyn_back]
    gt_dyn_back.sort()
    gt_dyn_back = gt_dyn_back[999:3999]

    baseline_mask = os.path.join(dataset_path, 'baseline', 'ROI.bmp')
    night_mask = os.path.join(dataset_path, 'night', 'ROI.bmp')
    dyn_back_mask = os.path.join(dataset_path, 'dynamic_background', 'ROI.bmp')
    cam_jitter_mask = os.path.join(dataset_path, 'camera_jittering', 'ROI.bmp')

    baseline_input_path = os.path.join(dataset_path, 'baseline', 'input')
    night_input_path = os.path.join(dataset_path, 'night', 'input')
    dyn_back_input_path = os.path.join(dataset_path, 'dynamic_background', 'input')
    cam_jitter_input_path = os.path.join(dataset_path, 'camera_jittering', 'input')

    input_baseline = os.listdir(baseline_input_path)
    input_night = os.listdir(night_input_path)
    input_cam_jitter = os.listdir(cam_jitter_input_path)
    input_dyn_back = os.listdir(dyn_back_input_path)

    input_baseline = [os.path.join(baseline_input_path, f) for f in input_baseline]
    input_baseline.sort()
    input_baseline = input_baseline[299:1199] # 1200
    input_night = [os.path.join(night_input_path, f) for f in input_night]
    input_night.sort()
    input_night = input_night[799:2999] # 3000
    input_cam_jitter = [os.path.join(cam_jitter_input_path, f) for f in input_cam_jitter]
    input_cam_jitter.sort()
    input_cam_jitter = input_cam_jitter[799:1199]
    input_dyn_back = [os.path.join(dyn_back_input_path, f) for f in input_dyn_back]
    input_dyn_back.sort()
    input_dyn_back = input_dyn_back[999:3999]

    input_files = [input_baseline, input_night, input_cam_jitter, input_dyn_back]
    lob_files = [baseline_lob, night_lob, cam_jitter_lob, dyn_back_lob]
    rcnn_files = [baseline_rcnn, night_rcnn, cam_jitter_rcnn, dyn_back_rcnn]
    gt_files = [gt_baseline, gt_night, gt_cam_jitter, gt_dyn_back]
    offsets = [300, 800, 800, 1000]
    order = ['baseline', 'night', 'camera jittering', 'dynamic background']
    masks = [baseline_mask, night_mask, cam_jitter_mask, dyn_back_mask]

    for i in range(0, len(lob_files)):
        lob_reader = LobsterReader(lob_files[i])
        rcnn_reader = RCNNReader(rcnn_files[i])
        gt_reader = GTReader(gt_files[i])

        lob_reader.read()
        rcnn_reader.read(i)
        gt_reader.extract_boxes(offsets[i], masks[i], i)
        tp_lob, fp_lob, fn_lob, tp_rcnn, fp_rcnn, fn_rcnn = get_statistics(lob_reader.boxes, rcnn_reader.boxes, gt_reader.gt, input_files[i], offsets[i])
        recall_lob = float(tp_lob) / float(tp_lob + fn_lob)
        precision_lob = float(tp_lob) / float(tp_lob + fp_lob)
        recall_rcnn = float(tp_rcnn) / float(tp_rcnn + fn_rcnn)
        precision_rcnn = float(tp_rcnn) / float(tp_rcnn + fp_rcnn)
        print('Results for {}'.format(order[i]))
        print('Recall LOBSTER = {}, Precision LOBSTER = {}'.format(recall_lob, precision_lob))
        print('Recall RCNN = {}, Precision RCNN = {}'.format(recall_rcnn, precision_rcnn))


if __name__ == '__main__':
    main()
