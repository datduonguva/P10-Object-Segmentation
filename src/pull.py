from pycocotools.coco import COCO
from pprint import pprint
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

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

def area_analysis(image_ids):
    """ 
    This is to understand the distribution of number of objects in each images
    """

    images = coco.loadImgs(image_ids)
    for image in images:
        annotation_ids = coco.getAnnIds(imgIds=image_json['id'], catIds=people_id, iscrowd=False)
        annotations = coco.loadAnns(annotation_ids)



if __name__ == "__main__":
    #plot_image(images[np.random.randint(len(images))])

raise SystemExit
for item in annotation:
    print("ratio: {}".format(item['area']/img['height']/img['width']))

#coco.showAnns(annotation)
#plt.show()


