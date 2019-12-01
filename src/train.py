from pycocotools.coco import COCO
from pycocotools import mask
from pprint import pprint
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
from keras.layers import BatchNormalization, LeakyReLU, Conv2D
from keras.layers import Conv2DTranspose, Input, Concatenate
from keras.applications import MobileNetV2
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.losses import binary_crossentropy
from keras.utils import plot_model 
data_dir = "../data"
val_type = "val2017"
train_type = 'val2017'

train_annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, train_type)
val_annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, val_type)

train_coco = COCO(train_annotation_file)
val_coco = COCO(val_annotation_file)

# only select images with name "person"
train_people_id = train_coco.getCatIds(catNms=['person'])
val_people_id = val_coco.getCatIds(catNms=['person'])

train_image_ids = train_coco.getImgIds(catIds=train_people_id)[:500]
val_image_ids = val_coco.getImgIds(catIds=val_people_id)[600:700]


def train_model():

    log_dir = 'logs/001/'
    logging = TensorBoard(log_dir=log_dir)

    df_train = area_filter(train_image_ids, train_coco)
    df_val = area_filter(val_image_ids, val_coco)

    train_generator = image_generator(df_train, train_coco, train_type)
    val_generator = image_generator(df_val, val_coco, val_type)



    checkpoint = ModelCheckpoint(log_dir + 'best_model.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    model = create_model()

    plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=np.ceil(len(train_image_ids)/16),
                        epochs=2,
                        callbacks=[checkpoint, logging],
                        validation_data=val_generator,
                        validation_steps=np.ceil(len(val_image_ids)/16))

def custom_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

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

def area_filter(image_ids, coco, threshold=0.1):
    """ 
    This is to filter out object that is too small.
    Return the dataframe containing the image_id and annotation_id
    """
    results = []
    images = coco.loadImgs(image_ids)
    for image in images:
        annotation_ids = coco.getAnnIds(imgIds=image['id'], catIds=train_people_id, iscrowd=False)
        annotations = coco.loadAnns(annotation_ids)
        for annotation in annotations:
            results.append([image['id'], annotation['id'], image['width'], image['height'], annotation['area']])
    
    df = pd.DataFrame(results, columns = ['image_id', 'annotation_id', 'width', 'height', 'segment_area'])
    df['ratio'] = df['segment_area']/df['width']/df['height']

    # only select the object whose segment area is > 10% of image area.
    df = df[df['ratio'] > threshold]

    print(df['ratio'].describe())

    return df

def process_mask(coco_object, annotation_ids, width=0, height=0):
    """
    This is to get the mask from annotation ids of an image
    """
    annotations = coco_object.loadAnns(annotation_ids)
    polygons = []
    for annotation in annotations:
        polygons += annotation['segmentation']
    encoded_mask = mask.frPyObjects(polygons, height, width) 

    # get the encoded mask
    binary_mask = mask.decode(encoded_mask)

    # decode to binary
    binary_mask = np.sum(binary_mask, axis=-1) > 0
    binary_mask = binary_mask.astype(np.float)
    
    # return in  H*W dimension
    return binary_mask

def preprocess(input_image, input_mask, input_size=128):
    """
    Resize image and mask to square of input_size
    Normalize images to [0.0, 1.0)
    """
    input_image_ = cv2.resize(input_image, (input_size, input_size))
    input_image_ = input_image_/255.0
    input_mask_ = cv2.resize(input_mask, (input_size, input_size))
    
    return input_image_, input_mask_

def image_generator(df, coco, data_type, batch_size=16):

    annotation_dict = df.groupby('image_id')['annotation_id'].apply(list).to_dict()
    image_id_list = np.array(list(annotation_dict.keys()))
    n = len(image_id_list)
    i = 0
    while True:
        input_images, input_masks = [], []
        for b in range(batch_size):
            if i == 0:          # reset a the end of the datasett
                np.random.shuffle(image_id_list)

            # get the image json containing the file_name
            image_json = coco.loadImgs([image_id_list[i]])[0]
            
            # load the image
            raw_image = cv2.imread(os.path.join("{}/{}/{}".format(data_dir, data_type, image_json['file_name'])))
            height, width, _ = raw_image.shape

            # decode the mask, using annotation id created at the group by above
            binary_mask = process_mask(coco, annotation_dict[image_id_list[i]], width, height)

            # preprocess input and mask (resize to 128, scale to [0, 1))
            input_image, input_mask = preprocess(raw_image, binary_mask)
            input_images.append(input_image)
            input_masks.append(input_mask)
            i = (i+1) % n
        input_images = np.array(input_images)
        input_masks = np.expand_dims(np.array(input_masks), axis=-1)

        yield input_images, input_masks

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
                    'block_16_project',      # 4x4
                ]

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

if __name__ == "__main__":
    train_model()
