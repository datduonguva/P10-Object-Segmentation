import cv2
import sys
import numpy as np

def average_contour(cnt):
    """
    This function smooths out the contours created by the model
    """
    result = [cnt[0]]
    n = len(cnt)
    for i in range(1, n):
        px = (cnt[(i-1)%n][0] + 2*cnt[i%n][0] + cnt[(i+1)%n][0])/4
        py = (cnt[(i-1)%n][1] + 2*cnt[i%n][1] + cnt[(i+1)%n][1])/4
        result.append([int(px), int(py)])

    return result

def find_contours(mask_255, threshold=100):
    """
    This function takes a gray-scale image of range [0, 255],
    return the contours of the white regions
    """

    ret, image= cv2.threshold(mask_255, threshold, 255,0)
   
    contours,hierarchy = cv2.findContours(image, 1, 2)
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    contours = [i for i in contours if len(i) > 20]
    
    result = []
    for cnt in contours:
        cnt = contours[0]
        epsilon = 0.001*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        result.append([[i[0], i[1]] for i in approx[:, 0]])
    
    # smooth out the contours:
    result = [average_contour(cnt) for cnt in result]
    return result

def draw_contours(image, contours):
    """
    This function draw the contours
    on the given image
    """
    # Draw:
    for cnt in contours:
        n = len(cnt)
        for i in range(n):
            image = cv2.line(image, tuple(cnt[i%n]), tuple(cnt[(i+1)%n]), (0, 255, 0), 5)

    return image

if __name__ == '__main__':
    """
    Test to see if it works as expected
    """
    image = cv2.imread('/home/datduong/Desktop/mask.png', cv2.IMREAD_GRAYSCALE)

    contours = find_contours(image, int(sys.argv[1]))
    image2 = cv2.imread('/home/datduong/Desktop/mask.png')
    image2 = draw_contours(image2, contours)
    cv2.imshow('image2', image2)
    cv2.waitKey() 
    
