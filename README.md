# Introduction
In this tutorial we will try to identify actors from images of Titanic movie.We will train our machine learning model to identify only kate winslet, leonardo dicaprio and rest of them as unknown.
![leonardo dicaprio and kate winslet](readme-images/leonardo-Kate.png "leonardo dicaprio and kate winslet")
![Unknow and kate winslet ](readme-images/unknown-kate.png "Unknow and kate winslet")
# Overview
We will learn face detection using opencv, deep learning and machine learning libraries.
* OpenCV is used to recognize faces in an image. It detects presence and location of a face in an image, but does not identify it.
* Use FaceNet deep learning model to computes 128 features that quantify a face
* Train a Support Vector Machine (SVM) on top of the features and perform classification
* Evaluate the performance of model using k fold cross validation
* Recognize faces in images
# Source Code
Source code the project can be accessed at
    
    https://github.com/balajich/titanic-face-recognition.git
    
# Project Structure

Directory|Description
---|---
Dataset| Contains train images of kate ,leonardo and unknown
recognize-images| Contains images that we need to detect faces
extract_train_evaluvate.py| Extract faces from train images,create a feature set, train model and evaluate its performance
recognize_faces.py| Recognize a face from a given input image
res10_300x300_ssd_iter_140000.caffemodel|  Face detector model from that detects presence and location of a face in an image, but does not identify it.
openface_nn4.small2.v1.t7| Create features from detected face image
model.pickle| Face recognition model
le.pickle| Encoded labels
notes.md| Contains list of dependent libraries that needs to installed


# Run 
Run scripts in below order
* extract_train_evaluvate.py
* recognize_faces.py
# Performance
Accuracy of current model is not great. You can improve its accuracy by adding more labelled train images 
Mean Accuracy: 56.67% (Standard Deviation: +/- 24.94%)
![leonardo dicaprio,kate winslet and Avengers](readme-images/avengers-kate-leonardo.png "leonardo dicaprio,kate winslet and Avengers")
# References
* I dont take any credit for this. I learned this entire implementation from article written by Adrian Rosebrock  in article https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/. The article covers thing in much deeper
* All the images are sourced from google
  