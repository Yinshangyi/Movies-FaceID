# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import tqdm
import imutils


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
imgLinks = []

for (i, imagePath) in tqdm.tqdm(enumerate(imagePaths)):

    try:
        # Extract the person name from the path
        name = imagePath.split('/')[-2]

        # Load the image and convert it to RGB
        image = cv2.imread(imagePath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_rgb = imutils.resize(image_rgb, width=300)

        boxes = face_recognition.face_locations(image_rgb, model='cnn')

        # # Compute the facial embeddings
        encodings = face_recognition.face_encodings(image_rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
            imgLinks.append(imagePath)

    except:
        print('Image Not Valid')

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings" : knownEncodings, 
        "names" : knownNames,
        "link" : imgLinks}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()

