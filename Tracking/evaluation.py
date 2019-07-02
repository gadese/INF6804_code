import os
import mot_metrics
import cv2
from mot_metrics import CustomBBox
from mot_metrics import Rectangle
from collections import defaultdict


def get_gt(filename):
    gt = defaultdict(list)
    min_frame = 100000
    max_frame = -10000
    with open(filename, 'r') as file:
        for line in file:
            split = line.split(',')
            frame = int(split[0])
            id = int(split[1])
            x_min = int(float(split[2]))
            y_min = int(float(split[3]))
            x_max = x_min + int(float(split[4]))
            y_max = y_min + int(float(split[5]))
            gt[frame].append(CustomBBox(id, Rectangle(x_min, y_min, x_max, y_max)))
            if frame > max_frame:
                max_frame = frame
            if frame < min_frame:
                min_frame = frame
    return gt, min_frame, max_frame


def get_hypo(filename):
    hypo = defaultdict(list)
    with open(filename, 'r') as file:
        for line in file:
            split = line.split(' ')
            frame = int(split[0])
            id = int(split[1])
            x_min = int(float(split[2]))
            y_min = int(float(split[3]))
            x_max = int(float(split[4]))
            y_max = int(float(split[5]))
            hypo[frame].append(CustomBBox(id, Rectangle(x_min, y_min, x_max, y_max)))
    return hypo


def visualize(frame, hypo, gt, folder, save=False):
    path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(path, 'detections', folder, 'frame' + str(frame) + '.jpg')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gt_boxes = gt[frame]
    boxes = hypo[frame]
    for box in gt_boxes:
        pt1 = (box.rectangle.x_tl, box.rectangle.y_tl)
        pt2 = (box.rectangle.x_br, box.rectangle.y_br)
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 4)
    for box in boxes:
        cv2.rectangle(img, (box.rectangle.x_tl, box.rectangle.y_tl), (box.rectangle.x_br, box.rectangle.y_br), (0, 0, 255), 4)
    if save:
        cv2.imwrite(folder + '_' + 'frame' + str(frame) + '.jpg', img)
    cv2.imshow('Vis', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    adl_rundle_gt = 'gt/gt_adl_rundle.txt'
    adl_rundle_hypo = 'track_results/adl_rundle.txt'
    adl_rundle = 'adl_rundle6'

    tud_stadtmitte_gt = 'gt/gt_tud_statdmitte.txt'
    tud_stadtmitte_hypo = 'track_results/tud_statdmitte.txt'
    tud = 'tud_stadmitte'

    venice_gt = 'gt/gt_venice.txt'
    venice_hypo = 'track_results/venice.txt'
    venice = 'venice2'

    gt = tud_stadtmitte_gt
    hypo = tud_stadtmitte_hypo
    folder = tud
    frame = 120

    annotations, min_frame, max_frame = get_gt(gt)
    hypotheses = get_hypo(hypo)

    # visualize(frame, hypotheses, annotations, folder, save=False)

    mot_metrics.main(hypotheses, annotations, min_frame, max_frame)


if __name__ == '__main__':
    main()