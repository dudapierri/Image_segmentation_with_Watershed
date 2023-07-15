import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt

directory = 'samples'
file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
for file_name in file_names:
    img = cv2.imread(f'{directory}/{file_name}')
    
    #rows_to_remove = list(range(500)) + list(range(3500, 4000))
    #cols_to_remove = list(range(250))
    #img = np.delete(img, rows_to_remove, axis=0)
    #img = np.delete(img, cols_to_remove, axis=1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray)

    median = cv2.medianBlur(gray, 9)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200))
    tophat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT, kernel)

    alpha = 3  # contrast control (1.0-3.0)
    beta = 75    # brightness control (0-100)
    contrast = cv2.convertScaleAbs(tophat, alpha=alpha, beta=beta)

    maxima = cv2.dilate(contrast, None, iterations=80)
    extended_maxima = cv2.subtract(maxima, contrast)

    _, extended_maxima = cv2.threshold(extended_maxima, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    extended_maxima = cv2.bitwise_not(extended_maxima)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(extended_maxima)
    mask = (stats[:, cv2.CC_STAT_AREA] >= 250).take(labels).astype(np.uint8)
    area_opening = cv2.bitwise_and(extended_maxima, extended_maxima, mask=mask)

    num_circles, circles, stats, centroids = cv2.connectedComponentsWithStats(area_opening)

    for i in range(1, num_circles):
        contours, _ = cv2.findContours((circles == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        perimeter = cv2.arcLength(contour, True)
        area = stats[i, cv2.CC_STAT_AREA]    
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.8:
            circles[circles == i] = 0

    circles[circles > 0] = 255
    circles = circles.astype(np.uint8)

    dist_transform = cv2.distanceTransform(circles, cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

    markers = cv2.connectedComponents(dist_transform.astype(np.uint8))[1]
    markers = cv2.watershed(cv2.cvtColor(circles, cv2.COLOR_BGR2RGB), markers)

    labels = np.unique(markers)[1:]
    colors = [ [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for label in labels ]

    overlaped = img.copy()

    height, width, _ = overlaped.shape
    for i in range(height):
        for j in range(width):
            label = markers[i,j]
            if label > 1:
                overlaped[i,j] = colors[label-1]

    overlaped = cv2.resize(overlaped, (round(height*0.2), round(width*0.2)))
    cv2.imshow(file_name, overlaped)
    cv2.waitKey(0)

cv2.destroyAllWindows()