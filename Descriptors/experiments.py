import os
import matplotlib.pyplot as plt
import seaborn as sns
from hog import HOG
from lbp import LBP
from datasets import Cifar10, DescribableTexture, Faces
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cifar_path = os.path.join('datasets', 'cifar-10')
    texture_path = os.path.join('datasets', 'texture')
    faces_path = os.path.join('datasets', 'faces')
    cifar10 = Cifar10(root, cifar_path)
    cifar10.read_data()
    cifar10.rotate_images()
    desc_texture = DescribableTexture(root, texture_path)
    desc_texture.split_data()
    faces = Faces(root, faces_path)
    faces.split_data()
    hog = HOG(show_images=False)
    lbp = LBP(show_images=False)

    datasets = [cifar10, desc_texture, faces]
    descriptors = [lbp, hog]

    for dataset in datasets:
        for descriptor in descriptors:
            print('--------------------------------------------------------')
            print('Results for {} on {}:'.format(descriptor.__class__.__name__, dataset.__class__.__name__))
            descriptor.apply(dataset.train_data)
            clf = LinearSVC()
            clf = clf.fit(descriptor.descriptor, dataset.train_labels)
            descriptor.apply(dataset.test_data)
            predictions = clf.predict(descriptor.descriptor)
            score = clf.score(descriptor.descriptor, dataset.test_labels)
            print('Score = {}'.format(score))
            print(classification_report(dataset.test_labels, predictions, target_names=dataset.labels_names))
            cm = confusion_matrix(dataset.test_labels, predictions)
            ax = plt.subplot()
            sns.heatmap(cm, annot=True)
            ax.set_xlabel('Predicted labels');
            ax.set_ylabel('True labels');
            ax.set_title('Confusion Matrix for {}'.format(descriptor.__class__.__name__));
            ax.xaxis.set_ticklabels(dataset.labels_names);
            ax.yaxis.set_ticklabels(dataset.labels_names);
            plt.show()
            print('--------------------------------------------------------')
            print('\n\n')


if __name__ == '__main__':
    main()