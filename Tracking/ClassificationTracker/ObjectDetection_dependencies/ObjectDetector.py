import numpy as np
import tensorflow as tf
from distutils.version import StrictVersion

from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import label_map_util


class ObjectDetector:
    def __init__(self, label_path = "ObjectDetection_dependencies/mscoco_label_map.pbtxt"):
        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = label_path
        self.detection_graph = None
        self.category_index = None

    def fetch_graph(self, frozen_path = "ObjectDetection_dependencies/SavedModels/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb"):
        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_path, 'rb') as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)

    def get_category_index(self):
        return self.category_index

    def find_item_id(self, item_id):
        # Trouver l'index des pommes ou autre élément d'intérêt
        for index, element in self.category_index.items():
            for cat_id, name in element.items():
                if name == item_id:
                    return index

    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict