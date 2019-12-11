import cv2
import numpy as np


def find_contours(mask_255):
    """
    This function takes a gray-scale image of range [0, 255],
    return the contours of the white regions
    """

    ret, image= cv2.threshold(mask_255, 150, 255,0)
    image = image.copy()

    # fill the small holes in the images
    t1 = (image[:-2, :] == 255)
    t2 = (image[1:-1, :] == 0)
    t3 = (image[2:, :] == 255)
    t4 = np.logical_and(t1 == t2, t1 == t3) + 0
    t5 = np.pad(t4, ((1, 1), (0, 0)), 'constant', constant_values=0)
    image[t5 == 1] = 255
    
    t1 = (image[:, :-2] == 255)
    t2 = (image[:, 1:-1] == 0)
    t3 = (image[:, 2:] == 255)
    t4 = np.logical_and(t1 == t2, t1 == t3) + 0
    t5 = np.pad(t4, ((0, 0), (1, 1)), 'constant', constant_values=0)
    image[t5 == 1] = 255

    flooded = cv2.floodFill(image.copy(), None, (0, 0), 255)
    flooded = 255 - flooded[1]

    hole_filled_image = np.logical_or(image> 150, flooded > 150)
    hole_filled_image= (hole_filled_image*255.0).astype(np.uint8)
    
    contours,hierarchy = cv2.findContours(hole_filled_image, 1, 2)
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    contours = [i for i in contours if len(i) > 20]
    
    result = []
    for cnt in contours:
        cnt = contours[0]
        epsilon = 0.001*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        result.append([[i[0], i[1]] for i in approx[:, 0]])
    
    return result

def draw_contours(image, contours):

    # Draw:
    for cnt in contours:
        for i in range(len(cnt) - 1):
            image = cv2.line(image, tuple(cnt[i]), tuple(cnt[i+1]), (0, 255, 0), 3)

    return image




if __name__ == '__main__':
    image = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

    contours = find_contours(image)

    draw_contours(image, contours)
