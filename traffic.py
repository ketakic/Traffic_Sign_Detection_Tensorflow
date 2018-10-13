import glob
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure, io, transform
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.optimizers import *
from keras.callbacks import *
import os
import numpy as np
import matplotlib as mp
from keras import backend as K
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Declaring global variables
training_image_dir = 'GTSRB/Final_Training/Images/'
# reading test set file names and true labels from the csv file using pandas package
test_set = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')
num_of_classes = 43
# fixing the image size of all the images in the data set to 36x36
image_size = 36
image_training_arr = []
labels_training_arr = []
image_training_test = []
label_training_test = []
batch_size = 150
nb_epoch = 2



# Function to center the image, which is a step of pre-processing images.
def center_image(img):
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    return img

# Function to pre-process images which will be applied on each image of the data set before training
def preprocess_img(img):

    #Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    img = center_image(img)

    img = transform.resize(img, (image_size, image_size))
    # roll color axis to axis 0
    img = np.rollaxis(img,-1)
    return img

# Function to define the conv-net structure
def cnn_model():

    # creating a sequential model which is a linear stack of layers
    model = Sequential()
    K.set_image_dim_ordering('th')
    # will create a 4 conv layer model with a couple of pooling layers
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(3, image_size, image_size), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    return model

def init_model():
    model = cnn_model()

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def test_model(model):
    # Testing model
    for file, classid in zip(list(test_set['Filename']), list(test_set['ClassId'])):
        image_path = os.path.join('GTSRB/Final_Test/Images/', file)
        image_training_test.append(preprocess_img(io.imread(image_path)))
        label_training_test.append(classid)

    # converting test images to numpy arrays
    test_image_np_arr = np.array(image_training_test)
    test_label_np_arr = np.array(label_training_test)


    class_pred = model.predict_classes(test_image_np_arr)

    # calculating accuracy.
    acc = np.sum(class_pred == test_label_np_arr)/np.size(class_pred)
    print("Accuracy on test set = {}".format(acc))

# PLACE WHERE PROGRAM STARTS
# Pre-processing images and storing images & labels in numpy arrays
training_image_paths = glob.glob(os.path.join(training_image_dir, '*/*.ppm'))
np.random.shuffle(training_image_paths)
print('Processing images...')

for path in training_image_paths:
    # reading images from 'path'
    img = preprocess_img(io.imread(path))
    # fetching class labels from folder names
    label = int(path.split('/')[-2])
    image_training_arr.append(img)
    labels_training_arr.append(label)

image_np_array = np.array(image_training_arr, dtype='float32')
# creating one-hot vector of true labels.
labels_np_array = np.eye(num_of_classes, dtype='uint8')[labels_training_arr]

model = init_model()

# Training model

model.fit(image_np_array, labels_np_array,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_split=0.2,
          shuffle=True,
          verbose=1)


plot_model(model, to_file='model.png')
test_model(model)
