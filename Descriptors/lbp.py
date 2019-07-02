import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


class LBP:
    def __init__(self, points=8, radius=1, method='uniform', show_images=False):
        self.points = points
        self.radius = radius
        self.method = method
        self.show_images = show_images
        self.descriptor = []

    def apply(self, image_list):
        self.descriptor = []
        for i, image in enumerate(image_list):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(image=image, P=self.points, R=self.radius, method=self.method)
            if self.show_images:
                fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
                ax[0].axis('off')
                ax[1].axis('off')
                ax[0].imshow(image_list[i])
                ax[1].imshow(lbp, cmap=plt.cm.gray)
                plt.show()
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.points + 3), range=(0, self.points + 2))
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-5)
            self.descriptor.append(hist)
