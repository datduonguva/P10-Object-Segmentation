from pprint import pprint
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import BatchNormalization, LeakyReLU, Conv2D
from keras.layers import Conv2DTranspose, Input, Concatenate
from keras.applications import MobileNetV2
import keras.backend as K

def show_performance(model):
    """ This function is to show some examples from validation data """
    val_image_ids_ = [i for i in val_image_ids]
    np.random.shuffle(val_image_ids_)

    df_val = area_filter(val_image_ids_, val_coco)
    image_id = df_val['image_id'].iloc[0]
    annotation_ids = df_val[df_val['image_id'] == image_id]['annotation_id'].tolist()

    image_json = val_coco.loadImgs([image_id])[0]
    raw_image = cv2.imread(os.path.join("{}/{}/{}".format(data_dir, val_type, image_json['file_name'])))
    height, width, _ = raw_image.shape

    # decode the mask, using annotation id created at the group by above
    binary_mask = process_mask(val_coco, annotation_ids, width, height)

    # preprocess input and mask (resize to 128, scale to [0, 1))
    input_image, input_mask = preprocess(raw_image, binary_mask)

    input_mask = np.expand_dims(input_mask, axis=-1)
    predicted_mask = model.predict(np.array([input_image]))[0]

    plt.figure(figsize=(20, 20))

    title = ['Input Image', 'True Mask', 'Predicted Mask']
    display_list = [input_image[:, :, ::-1], input_mask, predicted_mask]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def load_trained_model(log_dir):
    """ this function loads a train model at log_dir, whose name is "best_model.h5" """
    model = create_model()

    model.load_weights(os.path.join(log_dir, "best_model.h5"))
    return model
    

def plot_image(image_json, coco):
    annotation_ids = coco.getAnnIds(imgIds=image_json['id'], catIds=train_people_id, iscrowd=False)
    annotations = coco.loadAnns(annotation_ids)

    image = cv2.imread(os.path.join("{}/{}/{}".format(data_dir, data_type, image_json['file_name'])))

    for annotation in annotations:
        segnments = np.array(annotation['segmentation'][0]).astype(int)
        for i in range(len(segnments)//2-1):
            image = cv2.line(image, 
                             (segnments[i*2], segnments[2*i + 1]),
                             (segnments[i*2 + 2], segnments[2*i + 3]),
                             (255, 0, 0))
    cv2.imshow('image', image)
    cv2.waitKey()

def preprocess(input_image, input_mask, input_size=128):
    """
    Resize image and mask to square of input_size
    Normalize images to [0.0, 1.0)
    """
    input_image_ = cv2.resize(input_image, (input_size, input_size))
    input_image_ = input_image_/255.0
    input_mask_ = cv2.resize(input_mask, (input_size, input_size))
    
    return input_image_, input_mask_

def down_sample(x, filters, size, apply_batch_norm=True):
    x = Conv2D(filters=filters, kernel_size=size, strides=2, padding='same', use_bias=False)(x)

    if apply_batch_norm:
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

def up_sample(x, filters, size, apply_dropout=False):
    x = Conv2DTranspose(filters=filters, kernel_size=size, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if apply_dropout:
        x = Dropout(0.5)(x)
    x = LeakyReLU()(x)
    return x

def create_model(output_channel=1):
    base_model = MobileNetV2(input_shape=[128, 128, 3], include_top=False)
    layer_names = [
                    'block_1_expand_relu',   # 64x64
                    'block_3_expand_relu',   # 32x32
                    'block_6_expand_relu',   # 16x16
                    'block_13_expand_relu',  # 8x8
                    'block_16_project']      # 4x4

    layers = [base_model.get_layer(name).output for name in layer_names]
    down_stack = Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False


    inputs = Input(shape=[128, 128, 3])
    x = inputs

    # get the output of the down_sample layers
    skips = down_stack(x)

    x = skips[-1]
    skips = reversed(skips[:-1])
    
    for skip_layer, filter_ in zip(skips, [512, 256, 128, 64]):
        x = up_sample(x, filter_, 3)
        x = Concatenate()([x, skip_layer])
    
    x = Conv2DTranspose(output_channel, kernel_size=3, strides=2, padding='same', activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=x)

if __name__ == '__main__':
    model = create_model()
    model.load_weights("/home/datduong/gdrive/projects/P10-Object-Segmentation/logs/001/best_model.h5")

    K.set_learning_phase(0)

    while True:
        fn = input("Image path: ")
        image = cv2.imread(fn)
        input_, _ = preprocess(image, image)
        output_ = model.predict(np.array([input_]))[0]
        output_ = output_[:, :, 0]
        output_ = cv2.resize(output_, (image.shape[1], image.shape[0]))

        width = 600
        height = image.shape[0]/image.shape[1]*width
        height = int(height)
        #output_ = cv2.resize(output_, (width, height))
        output_ = (output_ > 0.5)
        #image = cv2.resize(image, (width, height))
        blur = cv2.GaussianBlur(image,(25, 25),0)
        image[np.logical_not(output_)] = 0
        blur[output_] = 0
        image = image + blur
        cv2.imshow('input', image)
        #cv2.imshow('output', output_)
        cv2.waitKey()

