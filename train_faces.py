# import the necessary packages
import argparse
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--data_link", required=True,
	help="path to facial features file")
ap.add_argument("--output_link", required=True,
	help="path to output model")
args = vars(ap.parse_args())

data = None

with open(args['data_link'],'rb') as data_file:
    data = pickle.load(data_file)

X = data['encodings']
y = data['names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=30)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

data_model = {'model' : model}

with open(args['output_link'], 'wb') as model_file:
    pickle.dump(data_model, model_file)

print('Training successful with an accuracy of {}%'.format(model.score(X_test, y_test)))