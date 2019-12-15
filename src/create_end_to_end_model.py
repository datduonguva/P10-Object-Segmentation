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
    ratio = tf.dtypes.cast(tf.math.reduce_min(128/tf.shape(input_)[:2]), tf.float32)
    
    new_shape = ratio*tf.dtypes.cast(tf.shape(input_)[:2], tf.float32)
    new_shape = tf.dtypes.cast(new_shape, tf.int32)

    tmp = tf.image.resize(input_, new_shape)

    # create padding
    pad_before = (128 - new_shape)//2
    pad_after = (128 - new_shape - pad_before)
    paddings = tf.concat([pad_before, pad_after], 0)
    paddings = tf.reshape(paddings, [2, 2])
    paddings = tf.transpose(paddings)
    paddings = tf.concat([paddings, [[0, 0]]], 0)
    tmp = tf.pad(tmp, paddings)

    # change dimension to the expected value
    tmp = tf.dtypes.cast(tf.expand_dims(tmp, axis=0)/255.0, tf.float32)

    # do prediction    
    output_ = model(tmp)
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
