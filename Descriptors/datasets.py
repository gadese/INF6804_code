import numpy as np
import os
import pickle
import cv2
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.utils import shuffle


class Dataset:
    def __init__(self, root, path):
        self.root = os.path.join(root, path)
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.labels_names = None

    def resize(self, rows, cols, images, show_images=False):
        print('Resizing images...')
        for i, image in enumerate(images):
            image = cv2.resize(image, (rows, cols))
            if show_images:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(images[i])
                ax[1].imshow(image)
                plt.show()
            images[i] = image
        return images


class Faces(Dataset):
    def __init__(self, root, path):
        super(Faces, self).__init__(root, path)
        self.train_data = list()
        self.train_labels = list()
        self.test_data = list()
        self.test_labels = list()
        self.labels_names = list()
        self.test_size_ratio = 0.25
        self.faces = ['Serena_Williams', 'Ariel_Sharon']

    def split_data(self):
        print('Loading Faces dataset...')
        for index, face in enumerate(self.faces):
            folder_name = os.path.join(self.root, face)
            images_names = os.listdir(folder_name)
            images = list()
            labels = list()
            for name in images_names:
                images.append(skio.imread(os.path.join(folder_name, name)))
                labels.append(index)
            test_size = int(len(images) * self.test_size_ratio)
            self.test_data.extend(images[0:test_size])
            self.train_data.extend(images[test_size:len(images)])
            self.test_labels.extend(labels[0:test_size])
            self.train_labels.extend(labels[test_size:len(labels)])
            self.labels_names.append(face)
        self.test_data, self.test_labels = shuffle(self.test_data, self.test_labels)
        self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)


class DescribableTexture(Dataset):
    def __init__(self, root, path):
        super(DescribableTexture, self).__init__(root, path)
        self.train_data = list()
        self.train_labels = list()
        self.test_data = list()
        self.test_labels = list()
        self.labels_names = list()
        self.test_size = 25
        self.textures = ['cracked', 'frilly', 'zigzagged']

    def split_data(self):
        print('Loading Describable Texture dataset...')
        for index, texture in enumerate(self.textures):
            folder_name = os.path.join(self.root, texture)
            images_names = os.listdir(folder_name)
            images = list()
            labels = list()
            for name in images_names:
                images.append(skio.imread(os.path.join(folder_name, name)))
                labels.append(index)
            self.test_data.extend(images[0:self.test_size])
            self.train_data.extend(images[self.test_size:len(images)])
            self.test_labels.extend(labels[0:self.test_size])
            self.train_labels.extend(labels[self.test_size:len(labels)])
            self.labels_names.append(texture)
        self.test_data, self.test_labels = shuffle(self.test_data, self.test_labels)
        self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)
        self.test_data = self.resize(64, 64, self.test_data, show_images=False)
        self.train_data = self.resize(64, 64, self.train_data, show_images=False)


class Cifar10(Dataset):
    def __init__(self, root, path):
        super(Cifar10, self).__init__(root, path)
        self.train_data = list()
        self.train_labels = list()
        self.test_data = list()
        self.test_labels = list()
        self.labels_names = list()
        self.test_size = 1000
        self.train_batch = 'data_batch_1'
        self.test_batch = 'test_batch'
        self.meta = 'batches.meta'

    def read_data(self):
        print('Loading CIFAR-10 dataset...')
        train_data = []
        train_labels = []
        filename = os.path.join(self.root, self.train_batch)
        with open(filename, 'rb') as file:
            info = pickle.load(file, encoding='bytes')
            images = info[b'data']
            for i in range(images.shape[0]):
                image = images[i]
                train_data.append(np.reshape(image, newshape=(32, 32, 3), order='F'))
            train_labels = info[b'labels']
        test_data = []
        test_labels = []
        filename = os.path.join(self.root, self.test_batch)
        with open(filename, 'rb') as file:
            info = pickle.load(file, encoding='bytes')
            images = info[b'data']
            for i in range(self.test_size):
                image = images[i]
                test_data.append(np.reshape(image, newshape=(32, 32, 3), order='F'))
            test_labels = info[b'labels'][0:self.test_size]
        label_names = []
        filename = os.path.join(self.root, self.meta)
        with open(filename, 'rb') as file:
            label_names = pickle.load(file, encoding='bytes')
            label_names = [label_name.decode('utf-8') for label_name in label_names[b'label_names']]
        for i, label in enumerate(train_labels):
            if label == 0 or label == 4 or label == 6 or label == 9:
                self.train_labels.append(label)
                self.train_data.append(train_data[i])
        self.labels_names = ['airplane', 'deer', 'frog', 'truck']
        for i, label in enumerate(test_labels):
            if label == 0 or label == 4 or label == 6 or label == 9:
                self.test_labels.append(label)
                self.test_data.append(test_data[i])


    def rotate_images(self, show_images=False):
        print('Rotating CIFAR-10 test images')
        for idx, img in enumerate(self.test_data):
            rot_angle = np.random.randint(0, 4) * 90.0
            r_img = ndimage.rotate(img, angle=rot_angle, mode='nearest')
            self.test_data[idx] = r_img
            if show_images:
                print('Rotation angle = {:.2f}'.format(rot_angle))
                fig, ax = plt.subplots(2)
                ax[0].imshow(img)
                ax[1].imshow(r_img)
                plt.show()
