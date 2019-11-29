from pycocotools.coco import COCO
from pprint import pprint
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

data_dir = "../data"
data_type = "train2017"
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

def area_analysis(image_ids):
    """ 
    This is to understand the distribution of number of objects in each images
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
    df = df[df['ratio'] > 0.1]

    plt.hist(df['ratio'], bins=100, rwidth=0.8)
    plt.show()
    print(df['ratio'].describe())

    return df

if __name__ == "__main__":
    #plot_image(images[np.random.randint(len(images))])
    area_analysis(image_ids)




