import cv2
import numpy as np
from time import time

image = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

if __name__ == '__main__':
    ret,image= cv2.threshold(image, 150, 255,0)
    t0 = time()
    image2 = image.copy()

    t1 = (image2[:-2, :] == 255)
    t2 = (image2[1:-1, :] == 0)
    t3 = (image2[2:, :] == 255)
    t4 = np.logical_and(t1 == t2, t1 == t3) + 0
    t5 = np.pad(t4, ((1, 1), (0, 0)), 'constant', constant_values=0)
    image2[t5 == 1] = 255
    
    t1 = (image2[:, :-2] == 255)
    t2 = (image2[:, 1:-1] == 0)
    t3 = (image2[:, 2:] == 255)
    t4 = np.logical_and(t1 == t2, t1 == t3) + 0
    t5 = np.pad(t4, ((0, 0), (1, 1)), 'constant', constant_values=0)
    image2[t5 == 1] = 255

    flooded = cv2.floodFill(image2.copy(), None, (0, 0), 255)
    flooded = 255 - flooded[1]

    hole_filled_image = np.logical_or(image2> 150, flooded > 150)
    hole_filled_image= (hole_filled_image*255.0).astype(np.uint8)
    
    contours,hierarchy = cv2.findContours(hole_filled_image, 1, 2)
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    print(contours)
    print(len(contours))
    for cnt in contours:
        print(cnt.shape)
    cnt = contours[0]
    epsilon = 0.005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    points = [(i[0], i[1]) for i in approx[:, 0]] 

    output = np.zeros((128, 128, 3)).astype(np.uint8)
    output[hole_filled_image>125] = 255

    
    for i in range(len(approx)-1):
        output = cv2.line(output, points[i],  points[i+1], (0, 255, 0), 1)
    cv2.imshow('image',output)
    cv2.waitKey()
