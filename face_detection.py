# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import glob
import numpy as np
from shutil import copyfile
import tqdm
import os
import pickle
from scipy.spatial import distance
import imutils
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree

class FaceRecognition():

    def __init__(self, model_link):
        self.model = None

        with open(model_link,'rb') as model_file:
            self.model = pickle.load(model_file)
            self.model = self.model['model']


    def get_boxes(self, image):
        boxes = face_recognition.face_locations(image, model='cnn')
        return boxes

    def get_embedding(self, image, boxes):
        # Compute the facial embeddings    
        embeddings = face_recognition.face_encodings(image, boxes)
        return embeddings

    def add_boxes(self, image, boxes, names):
        image_boxes = image.copy()
        # loop over the recognized faces
        for ((top, right, bottom, left),name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(image_boxes, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image_boxes, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
        return image_boxes


    def predict_names(self, embeddings):
        predicted_names = []
        distances = self.model.kneighbors(X=embeddings, n_neighbors=1, return_distance=True)
        index = 0
        for distance in distances[0]:
            
            if distance > 0.5:
                predicted_names.append('Unknown')
            else:
                embedding_reshape = np.expand_dims(embeddings[index], axis=0)
                name = self.model.predict(embedding_reshape)
                name = str(name[0])
                predicted_names.append(name)
            
        return predicted_names


    def facial_recognition(self, image):
        try:
            # Load the image and convert it to RGB
            image = imutils.resize(image, width=500)
        except:
            print("Can't resize image")
            return 'Error'

        image_boxes = image.copy()

        boxes = self.get_boxes(image)

        if boxes:
            embeddings = self.get_embedding(image, boxes)
            predicted_names = self.predict_names(embeddings)
            image_boxes = self.add_boxes(image, boxes, predicted_names)
        
        return image_boxes