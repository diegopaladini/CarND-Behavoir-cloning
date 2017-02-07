
# coding: utf-8

# # Behavoir Cloning project

# The first thinking was about this project it was how to treat it. In the previous projects I treated classification problems but in this case I treat the problem like a **regression problem**; In fact my target is a continue value so it is not assimilable to a class.
# 
# The second thinking was about the method to build my CNN. During the lessons I studied a lot of technincs about trasfering learning but the **lessons and paper by NVIDIA** was very interesting so I decided to replicate it.
# 
# The third thought was how to gather the data because I read a lot of post on CarND forum that the dataset provided by Udacity it is not enough. There are two ways to do it:
# - use image augmentation 
# - play more 
# 
# Naturally I used the second option because I passionate about car arcade videogame and second, but I guess more important, I want to gather data for the **recovery problem** in order to have a plan B if my car go over the lanes.
# 
# ### Data gathering 
# 
# The modus operandi that I use to gather the data was:
# - Play for 2 laps on test track
# - Play one laps on challenge track to go down the mountain and play one laps to rise up the mountain
# - Recording some recovery event along the 2 tracks

# ## Load data

# In[27]:

#Load the library needed for the project
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import cv2
import math
get_ipython().magic('matplotlib inline')


# In this snippet I'll load the list of the images and them targets. <br>
# The images list will be used from the fit generator to load only the images that the training needs according to the batch size.<br>
# On targets for the left and right camera I apply an offset to the steering angle in order to normalize the perspective of these cameras <br>
# 
# ** Left offset camera  0.15 **<br>
# ** Right offset camera -0.15 **

# In[28]:

#Reading the target

f = open("driving_log.csv")

X = []
target = []

LEFT_OFFSET = 0.15
RIGHT_OFFSET = -0.15

reader = csv.reader(f, delimiter = ",")
for i,line in enumerate(reader):
    X.append(line[0].strip())
    X.append(line[1].strip())
    X.append(line[2].strip())
    angle = line[3].strip()
    angle = float(angle)
    #center
    target.append(angle)
    #left
    target.append(str(angle + LEFT_OFFSET))
    #right
    target.append(str(angle + RIGHT_OFFSET))
    
X_train = np.array(X)
y_train = np.array(target)
print(X_train.shape)
print(y_train.shape)


# ### Create train, validation and shuffle the data

# In this snippet I'll create the train an validation set with sklearn function and in the end I'll shuffle the result in order to avoid any "data ordering" effect on the datasets

# In[29]:

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=33)

print("Train shape : {}".format(X_train.shape))
print("Validation shape: {}".format(X_val.shape))

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train) 
X_val, y_val = shuffle(X_train, y_train) 


# ## Pre-Processing Model training

# This snippet contains the definition of the pre-processing functions.<br>
# According to the NVIDIA paper I'll **resize** the image to a size of (66 x 200 x 3) and to avoid the "Pixel art effect" I used the **interpolation technics** from OpenCV library<br>
# To reduce the useless data I'll **crop** the sky and the bottom part of the image; See image below
# 
# <img src="crop_example.jpg"></img>
# 

# In the end I'll perform ** data normalization ** on color channels with the range  ** [-1 , 1] ** in order to have:<br>
# ** MEAN ~ 0 **<br>
# ** VARIANCE ~ 1 **

# In[ ]:

def normalize_color(image_data):
    a = -1.
    b = 1.
    c_min = 0
    c_max = 255
    return a + ( ( (image_data - c_min)*(b - a) )/( c_max - c_min ) )

def crop_img(image):
    return image[math.floor(image.shape[0]/4):image.shape[0]-25, 0:image.shape[1]]

def read_img(path):
    return normalize_color(cv2.resize(crop_img(mpimg.imread(path.strip())),
                                      (200, 66), interpolation=cv2.INTER_AREA))

def process_images(images):
    current_train = [read_img(path) for path in images]
    return current_train


# ## Model Training

# Before to start the training of my CNN I have to manage the large amount of the data in memory, so I used a functionality of Keras framework named **fit generator**. With this functionality I can load my data in batch mode as my model needs and my laptop say thanks :)
# 
# ### Traning phase
# 
# The training phase was a good part of the job because I tried to reproduce the NVIDIA architecture and plus I add some **dropout layers** in order to prevent the over fitting of my network. A tricky part was to choose the correct values for my dropout layers and to choose the right activation function in order to introduce a good generalization.<br>
# 
# The values of dropout layer come from a lot of testing; For the first one after convolution I choose **0.5** and for the second layers before the last neuron I choose **0.2**
# 
# About the activation function first I tried with **ELU** and **TANH** because I read from the forum that them were better for this type of problem, but from my personal opinion the **RELU** perform better.
# 
# The **BATCH SIZE** is set to 128
# 
# For the **Optimization** I choose **ADAM** because manage the adaptive learning rate and also have an **exponational decay**
# 
# For the **Loss and Metric** I choose **Mean Squared error** because I have to treat a regression problem
# 
# About the number of the **epochs** I saw that 3 it is a good trade off beetween time and performance 
# 
# Below you'll find the final picture of my CNN for behavoir cloning
# 
# <img src="final_architecture.jpg"> 

# In[40]:

import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D



def generate_arrays_from_file(images, labels, batch_size):
    start = 0
    end = start + batch_size
    n = len(images)
    while True:
        X = np.array(process_images(images[start:end]), dtype=np.float32) 
        steerings = np.array(labels[start:end], dtype=np.float32)
        start += batch_size
        end += batch_size
        if start >= n:
            start = 0
            end = batch_size
        yield (X, steerings)

model = Sequential()
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), input_shape=(66, 200, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

datagen = generate_arrays_from_file(X_train, y_train, 128)


# Below you can find a summary of my model with a Keras report.<br>
# I decided to resize and crop my images in order to have a more little network and to speed up the training phase

# In[44]:

model.summary()


# In[42]:

model.fit_generator(datagen, samples_per_epoch=len(X_train), nb_epoch=3)


# ## Save the model
# 
# Here I save the weight of my model and export the architecture like a json file

# In[45]:

from keras.models import load_model
model.save("project3_weight.h5")


# In[46]:

json_string = model.to_json()

import json
with open('project3_model.json', 'w') as outfile:
    json.dump(json_string, outfile)


# ## Evaluate the model
# 
# As we see below the model have a little mse error. Usually the dropout technics mixed with adam optimizer it is a good way to avoid the overfitting but maybe an improvement can be to perform the k-CROSS validation.

# In[47]:

evaluate_gen = generate_arrays_from_file(X_val, y_val, 128)

model.evaluate_generator(evaluate_gen, 128)


# ## Play the game
#  
# After this part I modified the drive.py which include the normalization part of the data that come from the simulator.<br>
# I run with the car across the first track for a lot of laps it was well good and it was very great!
# 
# Thanks for reading,
# 
# Diego Paladini
