import argparse
import base64
import json
import cv2
import math

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import load_model

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def crop_img(image):
    return image[math.floor(image.shape[0]/4):image.shape[0]-25, 0:image.shape[1]]

def normalize_color(image_data):
    a = -1.
    b = 1.
    c_min = 0
    c_max = 255
    return a + ( ( (image_data - c_min)*(b - a) )/( c_max - c_min ) )

def normalize_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return normalize_color(cv2.resize(crop_img(image), (200, 66), interpolation=cv2.INTER_AREA))

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = normalize_image(image_array)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.15
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)
    
    #W.write(str(steering_angle))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition h5. Model should be on the same path.')
    args = parser.parse_args()

    model = load_model(args.model)
    
    #global W 
    #W = open('C:/Users/paladini_di/projects_repo/data_behavoir_cloning/out.csv', 'w')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)