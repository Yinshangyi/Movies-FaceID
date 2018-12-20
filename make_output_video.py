import cv2
import numpy as np
import tqdm
import imutils
from face_detection import FaceRecognition
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--input_link", required=True,
	help="path to input directory of faces + images")
ap.add_argument("--output_link", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("--model_link", required=True,
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

video_link = args['input_link']
output_video = args['output_link']

print('Processing the video ...')
stream = cv2.VideoCapture(video_link)
writer = None

# Get the number of frame
nb_frame = 0
last_frame = None

facialRec = FaceRecognition(args['model_link'])

while True:
    (grab, frame) = stream.read()

    if grab:
        last_frame = frame

    if not grab:
        break
    
    nb_frame += 1

print('There {} frame'.format(nb_frame))

stream = cv2.VideoCapture(video_link)

resize_frame = imutils.resize(last_frame, width=500)
resize_frame = np.array(resize_frame)
frame_width = resize_frame.shape[1]
frame_height = resize_frame.shape[0]

# Write the image to the output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(output_video, 
                        fourcc, 
                        24, 
                        (frame_width, frame_height))

for n in tqdm.tqdm(range(nb_frame)):
    # Get the next frame
    (grab, frame) = stream.read()

    # If no next frame, we have reached the end of the video
    if not grab:
        print('problem')
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predict_image = facialRec.facial_recognition(image_rgb)
    predict_image_bgr = cv2.cvtColor(predict_image, cv2.COLOR_RGB2BGR)

    writer.write(predict_image_bgr)

stream.release()
writer.release()