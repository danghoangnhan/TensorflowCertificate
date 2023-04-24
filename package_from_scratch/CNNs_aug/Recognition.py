# -*- coding: utf-8 -*-
"""
Created on Tue May 19 03:19:05 2020

@author: tuyen
"""


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from lee.preprocessing import ImageToArrayPreprocessor
#from lee.preprocessing import SimplePreprocessor
#from lee.datasets import SimpleDatasetLoader
#from lee.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from tensorflow.keras.models import load_model

#########################ImageToArrayPreprocessor##############################
# imports the img_to_array function from Keras
from keras.preprocessing.image import img_to_array
class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat
    def preprocess(self, image):
        # apply the the Keras utility function that correctly rearranges
        # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)
########################SimplePreprocessor####################################
import cv2
class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)
########################SimpleDatasetLoader###################################
#import numpy as np
#import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))
################################Recognition###################################

#classLabels = ["cat", "dog", "panda"]

classLabels = ["bluebell", "buttercup", "colts'foot", 
              "cowslip", "crocus", "daffodil", "daisy", 
              "dandelion", "fritillary", "iris",
              "lilyvalley", "pansy", "snowdrop",
              "sunflower", "tigerlily", "tulip",
              "windflower"]

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
#magePaths = np.array(list(paths.list_images(args["dataset"])))
imagePaths = list(paths.list_images("D:/Hop tac voi Indonesia 2022/UNDIKSHA/Programs/Flowers/test"))#
#idxs = np.random.randint(0, len(imagePaths), size=(15,))
#imagePaths = imagePaths[idxs]
# initialize the image preprocessors
#sp = SimplePreprocessor(299,299)
# For KdadexNET
#sp = SimplePreprocessor(32,32)
# For LeNet
sp = SimplePreprocessor(28,28)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0
print(data.shape)
# load the pre-trained network
print("[INFO] loading pre-trained network...")

#model = load_model("D:/Hop tac voi Indonesia 2022/UNDIKSHA/weights/shallownet_weights_new.hdf5")
#model = load_model("D:/Hop tac voi Indonesia 2022/UNDIKSHA/weights/ResNet_weights.hdf5")
model = load_model("D:/Hop tac voi Indonesia 2022/UNDIKSHA/weights/myNET_weights.hdf5")
# make predictions on the images
print("[INFO] recognition...")
preds = model.predict(data, batch_size=32).argmax(axis=1)
# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
# load the example image, draw the prediction, and display it
# to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    
    save_path = "D:/Hop tac voi Indonesia 2022/UNDIKSHA/results/"+str(i)+".bmp"
    cv2.imwrite(save_path, image)
    #print("E:/MCUT/Neural Networks/Day 4/1.ShallowNet/recognition_results/"+str(i)+".bmp")
    cv2.waitKey(0)