# import the necessary packages
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG19
from keras.datasets import mnist
import numpy as np
import argparse
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_mnist = x_train[0]
label = y_train[0]

image = cv2.resize(image_mnist, (224, 224)) # resize image to 224x224
image = cv2.merge((image,image,image)) # convert to 3 channels
# cv2.imshow("Image", image)
# cv2.waitKey(0)

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
# args = vars(ap.parse_args())

# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pixels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
# print("[INFO] loading and preprocessing image...")
# image = load_img(args["image"], target_size=(224, 224))
# image = img_to_array(image_mnist)

# our image is now represented by a NumPy array of shape (224, 224, 3),
# assuming TensorFlow "channels last" ordering of course, but we need
# to expand the dimensions to be (1, 3, 224, 224) so we can pass it
# through the network -- we'll also preprocess the image by subtracting
# the mean RGB pixel intensity from the ImageNet dataset
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# load the VGG16 network
print("[INFO] loading network...")
model = VGG19(weights="imagenet")

# classify the image
print("[INFO] classifying image...")
preds = model.predict(image)
P = decode_predictions(preds)

prediction = P[0][1]
print(prediction)

# loop over the predictions and display the rank-5 predictions +
# # probabilities to our terminal
# for (i, (imagenetID, label, prob)) in enumerate(P[0]):
#     print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))




