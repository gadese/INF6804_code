# INF6804 - Computer Vision

This repository hosts our code written as part of our Computer Vision class. 


It contains three module, one for each of the practical work assignments we had.

1. Background segmentation: This first assignment aimed to compare a traditional background segmentation method with a modern CNN method. With this in mind, we used:
	..1. Lobster algorithm
	..2. Tensorflow Object Detection API with Faster RCNN
	..3. The comparison was done for different use cases (Baseline, low-light/contrast, dynamic background, and with a jittering camera)
	
2. Descriptors: This lab was about comparing two different descriptors to obtain the main features of an image:
	..1. History of Oriented Gradients (HOG)
	..2. Local Binary-Pattern (LBP)
	..3. The descriptors were used in various use cases(Textures, faces, and CIFAR-10 classes) with a LinearSVC classifier.

	
3. Tracking: This lab was a major assignment about object tracking in a video. The goal was to follow two cups in a video. We used:
	..1. Tensorflow Object Detection API with Faster RCNN (used as a classification tracker)
	..2. OpenCV's built-in trackers
	..3. Compared the results from both on various datasets from MOT challenge
	..4. Tested our tracker with the given video sequence
	
	
This lab was completed with @beaupreda.
