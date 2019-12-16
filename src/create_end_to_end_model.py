import os
from keras.models import Model
import cv2
import tensorflow as tf
from train import create_model
import keras.backend as K
import numpy as np
from keras.layers import Input
def create_end_to_end_tf_model(log_dir):
    model = create_model()
    model.load_weights(os.path.join(log_dir, "best_model.h5"))
    K.set_learning_phase(0)


    input_ = tf.placeholder(tf.float32, (128, 128, 3))

    # do prediction    
    x = tf.expand_dims(input_, axis=0)
    output_ = model(x)
    sess = K.get_session()
    
    tf.saved_model.simple_save(sess,
                               os.path.join(log_dir, "serve"),
                               inputs={'input': input_},
                               outputs={'output': output_})
def convert_to_tf(log_dir):
    create_end_to_end_tf_model(log_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(log_dir, "serve"))
    tflite_model = converter.convert()
    with open(os.path.join(log_dir, "best_model.tflite"), "wb") as f:
        f.write(tflite_model)

if __name__ == '__main__':
    log_dir = r"/home/datduong/gdrive/projects/P10-Object-Segmentation/logs/004"
    create_end_to_end_tf_model(log_dir)
