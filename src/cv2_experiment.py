import cv2
import numpy as np
from time import time

image = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

def _flood(image, flooded, i, j, start_i, start_j):
    if i == start_i and j == start_j:
        flooded[i, j] = 1
    elif image[i, j] < 150:
        for ni, nj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            new_i = i + ni
            new_j = j + nj
            if new_i < 0 or new_i > 127 or new_j < 0 or new_j > 127:
                continue
                 
            if flooded[new_i, new_j] == 1:
                flooded[i, j] = 1
                break
def flood_fill(image):
    flooded = np.zeros(image.shape)
    for i in range(128):
        for j in range(128):
            _flood(image, flooded, i, j, 0, 0)
    for i in range(127, -1, -1):
        for j in range(127, -1, -1):
            _flood(image, flooded, i, j, 127, 127)
    for i in range(128):
        for j in range(127, -1, -1):
            _flood(image, flooded, i, j, 0, 127)
    for i in range(127, -1, -1):
        for j in range(128):
            _flood(image, flooded, i, j, 127, 0)

    return flooded
if __name__ == '__main__':
#    flooded = flood_fill(image)
#    flooded = (flooded*255).astype(np.uint8)
#
    original_image = cv2.imread('/home/datduong/Desktop/hinh2.png', cv2.IMREAD_GRAYSCALE)

    ret,thresh = cv2.threshold(original_image,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    print(contours)
    print(len(contours))
    cnt = contours[2]
    epsilon = 0.005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    points = [(i[0], i[1]) for i in approx[:, 0]] 

    original_image = cv2.imread('/home/datduong/Desktop/hinh2.png')
#    for i in range(len(approx)-1):
#        original_image = cv2.line(original_image, points[i],  points[i+1], (255,0, 0), 2)
    original_image = cv2.drawContours(original_image, contours, 2, (0, 255, 0), 1)
    cv2.imshow('image', original_image)
    cv2.waitKey()
