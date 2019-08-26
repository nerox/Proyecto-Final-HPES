from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import cv2
import pickle
import pandas as pd
import json
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

IMG_SIZE=128

def load_first_model():
    model = Sequential()

    pickle_in = open("Features_output.pickle","rb")
    X = pickle.load(pickle_in)
    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    checkpoint_path = "/home/group04/Documents/proyecto3/Proyecto-Final-HPES/jetson/modelo_ale/cp.ckpt"

    model.load_weights(checkpoint_path)
    return model

def get_image_in_format(filepath):      
    im = np.array(Image.open(filepath))
    print(im.shape)
    new_array = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 

def get_image(filepath):
    im = np.array(Image.open(filepath))
    return im
def main():
    first_model=load_first_model()
    #agregar funcionalidad de caputrar imagen.
    #por ahora estamos usando una imagen fija
    plate_image=get_image_in_format('/home/group04/Documents/proyecto3/Proyecto-Final-HPES/jetson/model_inputs/prueba10.jpg')
    raw_image=get_image('/home/group04/Documents/proyecto3/Proyecto-Final-HPES/jetson/model_inputs/plate1.jpeg')
    print("hola")
    prediction_upon_image_captured=first_model.predict(plate_image)
    print("hola2")
    if prediction_upon_image_captured==0:
        print("Es una placa, hora de buscar la ultima letra")
        #forma burda de recortar la imagen pra obtener la ultima posicion de la placa
        x_point_top = raw_image.shape[1]/8*6
        y_point_top = 0
        x_point_bot = raw_image.shape[1]
        y_point_bot = raw_image.shape[0]
        plateImage = Image.fromarray(raw_image).crop((x_point_top, y_point_top, x_point_bot, y_point_bot))
        plateImage=plateImage.resize((IMG_SIZE,IMG_SIZE))
        #deberia hacer un llamado al modelo de verny
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        # Set title
        ax.set_title('Output Image')
        # Display the images
        ax.imshow(plateImage)
        plt.show()
    else:
        print("No es placa")
       
    
  
if __name__== "__main__":
  main()
