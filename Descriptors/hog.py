import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.exposure import exposure


class HOG:
    def __init__(self, orientation=8, pixels_per_cells=(8, 8), cells_per_block=(4, 4), block_norm='L2',
                 feature_vector=True, show_images=False):
        self.orientation = orientation
        self.pixels_per_cells = pixels_per_cells
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.feature_vector = feature_vector
        self.show_images = show_images
        self.descriptor = []

    def apply(self, image_list):
        self.descriptor = []
        for i, image in enumerate(image_list):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            desc, hog_image = None, None
            if self.show_images:
                desc, hog_image = hog(image=image, orientations=self.orientation, pixels_per_cell=self.pixels_per_cells,
                                  cells_per_block=self.cells_per_block, block_norm=self.block_norm,
                                  visualize=self.show_images, feature_vector=self.feature_vector)
                fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
                ax[0].axis('off')
                ax[1].axis('off')
                hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                ax[0].imshow(image_list[i])
                ax[1].imshow(hog_image, cmap=plt.cm.gray)
                plt.show()
            else:
                desc = hog(image=image, orientations=self.orientation, pixels_per_cell=self.pixels_per_cells,
                           cells_per_block=self.cells_per_block, block_norm=self.block_norm,
                           visualize=self.show_images, feature_vector=self.feature_vector)
            self.descriptor.append(desc)
