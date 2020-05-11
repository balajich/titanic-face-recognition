# USAGE
# python extract_train_evaluvate.py
# import the necessary packages
import argparse
import os
import pickle

import cv2
import imutils
import numpy
import numpy as np
from imutils import paths
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Read arguments passed to script
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-i", "--dataset", default='dataset',
                             help="path to input directory of faces")
args = vars(argument_parser.parse_args())

# fix random seed for reproducibility - it allows that no matter if we execute
# the code more than one time, the random values have to be the same
seed = 7
np.random.seed(seed)

# Reading face detector model from the disk
print("Reading face detector model from the disk")
face_detector_model = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Reading face embedding model from the disk
print("Reading face embedding model from the disk")
face_embedding_model = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# Read train dataset of images
print("Read train dataset of images")
imagePaths = list(paths.list_images(args["dataset"]))

train_embeddings = []
train_names = []

# initialize the total number of faces processed
total_no_of_faces_processed = 0

# Iterate through train dataset
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    face_name = imagePath.split(os.path.sep)[-2]
    print(" processing image {}/{} - {}".format(i + 1,
                                               len(imagePaths), face_name))

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    face_detector_model.setInput(imageBlob)
    detections = face_detector_model.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            face_embedding_model.setInput(faceBlob)
            vec = face_embedding_model.forward()

            # add the name of the person + corresponding face
            # embedding to their respective lists
            train_names.append(face_name)
            train_embeddings.append(vec.flatten())
            total_no_of_faces_processed += 1

data_set = {"embeddings": train_embeddings, "names": train_names}

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data_set["names"])
X = np.array(data_set["embeddings"])
y = np.array(labels)

print('Train and evaluate the model')
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for (train, test) in kfold.split(X, y):
    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("training model...")
    recognizer_model = SVC(C=1.0, kernel="linear", probability=True)
    recognizer_model.fit(X[train], y[train])
    y_pred = recognizer_model.predict(X[test])
    accuracy = accuracy_score(y[test], y_pred)
    print('Accuracy Score :', accuracy)
    cvscores.append(accuracy * 100)

print("Mean Accuracy: %.2f%% (Standard Deviation: +/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

# write the actual face recognition model to disk
f = open('model.pickle', "wb")
f.write(pickle.dumps(recognizer_model))
f.close()

# write the label encoder to disk
f = open('le.pickle', "wb")
f.write(pickle.dumps(le))
f.close()
