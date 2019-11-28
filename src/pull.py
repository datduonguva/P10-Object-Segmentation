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

images = coco.loadImgs(image_ids)

a = {}
for image in images:
    annotation_ids = coco.getAnnIds(imgIds=image['id'], catIds=people_id, iscrowd=None)
    len_ = len(annotation_ids)
    if len_ not in a:
        a[len_] = 1
    else:
        a[len_] += 1
print(a)

raise SystemExit
annotation = coco.loadAnns(annotation_id)
for item in annotation:
    print("ratio: {}".format(item['area']/img['height']/img['width']))

#coco.showAnns(annotation)
#plt.show()


