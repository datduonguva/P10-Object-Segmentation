from pycocotools.coco import COCO
from pycocotools import mask
from pprint import pprint
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

data_dir = "../data"
data_type = "val2017"
annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
coco = COCO(annotation_file)

# print categories
cats = coco.loadCats(coco.getCatIds())

# only select images with name "person"
people_id = coco.getCatIds(catNms=['person'])

image_ids = coco.getImgIds(catIds=people_id)


def plot_image(image_json):
    annotation_ids = coco.getAnnIds(imgIds=image_json['id'], catIds=people_id, iscrowd=False)
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

def area_filter(image_ids, threshold=0.1):
    """ 
    This is to filter out object that is too small.
    Return the dataframe containing the image_id and annotation_id
    """
    results = []
    images = coco.loadImgs(image_ids)
    for image in images:
        annotation_ids = coco.getAnnIds(imgIds=image['id'], catIds=people_id, iscrowd=False)
        annotations = coco.loadAnns(annotation_ids)
        for annotation in annotations:
            results.append([image['id'], annotation['id'], image['width'], image['height'], annotation['area']])
    
    df = pd.DataFrame(results, columns = ['image_id', 'annotation_id', 'width', 'height', 'segment_area'])
    df['ratio'] = df['segment_area']/df['width']/df['height']

    # only select the object whose segment area is > 10% of image area.
    df = df[df['ratio'] > threshold]

    plt.hist(df['ratio'], bins=100, rwidth=0.8)
#    plt.show()
    print(df['ratio'].describe())

    return df

def process_mask(coco_object, annotation_ids):
    annotations = coco.loadAnns(annotation_ids)
    polygons = []
    for annotation in annotations:
        polygons += annotation['segmentation']
    encoded_mask = mask.frPyObjects(polygons, df.iloc[idx]['height'], df.iloc[idx]['width'])

    # get the encoded mask
    binary_mask = mask.decode(encoded_mask)

    # decode to binary
    binary_mask = np.sum(binary_mask, axis=-1) > 0
    binary_mask = binary_mask.astype(np.float)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    
    # return in  H*W*1 dimension
    return binary_mask

if __name__ == "__main__":
    #plot_image(images[np.random.randint(len(images))])
    df = area_filter(image_ids)
    idx = np.random.randint(len(df))
    i = df['image_id'].iloc[idx]
    annotation_ids = df[df['image_id'] == i]['annotation_id'].tolist()
    
