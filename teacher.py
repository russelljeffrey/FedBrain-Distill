import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from keras.layers import Flatten

vgg16 = VGG16(weights='imagenet', include_top=False)

def create_teacher_model():
    input_tensor = layers.Input(shape=(224, 224, 3))
    x = vgg16(input_tensor)
    x = Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output)
    return model
