# USAGE
# python recognize_faces.py

# import the necessary packages
import argparse
import pickle

import cv2
import imutils
import numpy as np

# construct the argument parser and parse the arguments
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-i", "--image", default='./recognize-images/kate_leonardo.jpg',
                             help="path to input image")
args = vars(argument_parser.parse_args())

# Reading face detector model from the disk
print("Reading face detector model from the disk")
face_detector_model = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Reading face embedding model from the disk
print("Reading face embedding model from the disk")
face_embedding_model = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# load the actual face recognition model along with the label encoder
recognizer_model = pickle.loads(open('model.pickle', "rb").read())
le = pickle.loads(open('le.pickle', "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(args["image"])
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

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections
    if confidence > 0.5:
        # compute the (x, y)-coordinates of the bounding box for the
        # face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # extract the face ROI
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            continue

        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
        face_embedding_model.setInput(faceBlob)
        vec = face_embedding_model.forward()

        # perform classification to recognize the face
        preds = recognizer_model.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # draw the bounding box of the face along with the associated
        # probability
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
# cv2.waitKey(0) # Wait for infinite time
cv2.waitKey(5 * 1000)  # Wait for 5 seconds
