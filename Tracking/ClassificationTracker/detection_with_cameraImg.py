import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir

from ClassificationTracker.object_detection.utils import visualization_utils as vis_util
from ClassificationTracker.ObjectDetection_dependencies.ObjectDetector import ObjectDetector

# adl_video = '../videos/ADL-Rundle-6.mp4'
# adl_file = '../detections/adl_rundle.txt'
# adl_save = '../detections/adl_rundle6/'
# IMAGE_SIZE = (19.2, 10.8)
# size = (1920, 1080)

# tud_video = '../videos/TUD-Stadtmitte.mp4'
# tud_file = '../detections/tud_stadtmitte.txt'
# tud_save = '../detections/tud_stadmitte/'
# IMAGE_SIZE = (6.4, 4.8)
# size = (640, 480)

venice_video = '../videos/Venice-2.mp4'
venice_file = '../detections/venice.txt'
venice_save = '../detections/venice2/'
IMAGE_SIZE = (19.2, 10.8)
size = (1920, 1080)

# videoPathTp3 = "../dataset/video.mp4"
# tp3_file = 'results_det.txt'
# tp3_save = '../results_detection/'
# IMAGE_SIZE = (19.2, 10.8)
# size = (1920, 1080)

cap = cv2.VideoCapture(venice_video)

obj_det = ObjectDetector()
obj_det.fetch_graph()
cup_id = obj_det.find_item_id('person')

image_list = []
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image_list.append(image[..., ::-1])


with open(venice_file, 'w') as myfile:
    i = 1
    VALEUR_SEUIL = 0.5
    for color in image_list:
        output_dict = obj_det.run_inference_for_single_image(color)
        category_index = obj_det.get_category_index()

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            color,
            output_dict['detection_boxes'][output_dict['detection_classes'] == cup_id, :],
            output_dict['detection_classes'][output_dict['detection_classes'] == cup_id],
            output_dict['detection_scores'][output_dict['detection_classes'] == cup_id],
            #output_dict['detection_boxes'],
            #output_dict['detection_classes'],
            #output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        DIM_X_IMAGE = color.shape[1]
        DIM_Y_IMAGE = color.shape[0]
        Ymin = output_dict['detection_boxes'][output_dict['detection_scores'] > VALEUR_SEUIL, 0] * DIM_Y_IMAGE
        Xmin = output_dict['detection_boxes'][output_dict['detection_scores'] > VALEUR_SEUIL, 1] * DIM_X_IMAGE
        Ymax = output_dict['detection_boxes'][output_dict['detection_scores'] > VALEUR_SEUIL, 2] * DIM_Y_IMAGE
        Xmax = output_dict['detection_boxes'][output_dict['detection_scores'] > VALEUR_SEUIL, 3] * DIM_X_IMAGE

        cup_mask = output_dict['detection_classes'] == cup_id
        x_max = []
        x_min = []
        y_max = []
        y_min = []
        j = 0
        for xin, xax, yin, yax in zip(Xmin, Xmax, Ymin, Ymax):
            if cup_mask[j]:
                x_min.append(xin)
                x_max.append(xax)
                y_min.append(yin)
                y_max.append(yax)
            j += 1

        for Xmi, Xma, Ymi, Yma in zip(x_min, x_max, y_min, y_max):
            myfile.write('{} {} {} {} {}\n'.format(i, int(Xmi), int(Ymi), int(Xma), int(Yma)))
            cv2.rectangle(color, (Xmi, Ymi), (Xma, Yma), (255, 0, 0), 4)

        cv2.imwrite('{}frame{}.jpg'.format(venice_save, i), color)
        i += 1
