import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir

from object_detection.utils import visualization_utils as vis_util
from ObjectDetection_dependencies.ObjectDetector import ObjectDetector

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

#Load saved images
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#TEST_IMAGE_PATHS = [ "object_detection/test_images/baseline/baseline/PETS2006/input/" + 'in000{}.jpg'.format(i) for i in range(300, 1000) ]
#TEST_IMAGE_PATHS += ["object_detection/test_images/baseline/baseline/PETS2006/input/" + 'in00{}.jpg'.format(i) for i in range(1000, 1200)]

#TEST_IMAGE_PATHS = [ "object_detection/test_images/nightVideos/nightVideos/streetCornerAtNight/input/" + 'in000{}.jpg'.format(i) for i in range(800, 1000) ]
#TEST_IMAGE_PATHS += ["object_detection/test_images/nightVideos/nightVideos/streetCornerAtNight/input/" + 'in00{}.jpg'.format(i) for i in range(1000, 5200)]

TEST_IMAGE_PATHS = [ "object_detection/test_images/dynamicBackground/dynamicBackground/fall/input/" + 'in00{}.jpg'.format(i) for i in range(1000, 4000) ]

#TEST_IMAGE_PATHS = [ "object_detection/test_images/cameraJitter/cameraJitter/sidewalk/input/" + 'in000{}.jpg'.format(i) for i in range(800, 1000) ]
#TEST_IMAGE_PATHS += ["object_detection/test_images/cameraJitter/cameraJitter/sidewalk/input/" + 'in00{}.jpg'.format(i) for i in range(1000, 1200)]


IMAGE_SIZE = (12, 8)

obj_det = ObjectDetector()
obj_det.fetch_graph()
apple_id = obj_det.find_item_id('apple')

image_list = []
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_list.append (load_image_into_numpy_array(image))

#sidewalk_ROI = cv2.imread("object_detection/test_images/cameraJitter/cameraJitter/sidewalk/ROI.bmp", 0)
with open('Fall_results/fall_boxes.txt', 'a') as myfile:

    i = 1000
    for color in image_list:
        #color = color[87:224, 0:189]
        output_dict = obj_det.run_inference_for_single_image(color)
        category_index = obj_det.get_category_index()

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            color,
            # output_dict['detection_boxes'][output_dict['detection_classes'] == apple_id, :],
            # output_dict['detection_classes'][output_dict['detection_classes'] == apple_id],
            # output_dict['detection_scores'][output_dict['detection_classes'] == apple_id],
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)

        DIM_X_IMAGE = color.shape[1]
        DIM_Y_IMAGE = color.shape[0]
        Ymin = output_dict['detection_boxes'][output_dict['detection_scores'] > 0.5 , 0] * DIM_Y_IMAGE
        Xmin = output_dict['detection_boxes'][output_dict['detection_scores'] > 0.5, 1] * DIM_X_IMAGE
        Ymax = output_dict['detection_boxes'][output_dict['detection_scores'] > 0.5, 2] * DIM_Y_IMAGE
        Xmax = output_dict['detection_boxes'][output_dict['detection_scores'] > 0.5, 3] * DIM_X_IMAGE

        ax = plt.axes()
        plt.imshow(color)
        #plt.savefig('streetCornerAtNight_results/streetCornerAtNight_image{}.jpg'.format(i))
        myfile.write('ID ' + str(i))
        for Xmi, Xma, Ymi, Yma in zip(Xmin, Xmax, Ymin, Ymax):
            myfile.write(' Xmin ' + str(Xmi) + ' Xmax ' + str(Xma) + ' Ymin ' + str(Ymi) + ' Ymax ' + str(Yma))
        myfile.write('\n')
        #plt.show()
        plt.close()
        i += 1







