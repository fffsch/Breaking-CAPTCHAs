import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

from pathlib import Path

data_dir = Path("images/train")



images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split("-0.png")[0] for img in images]

characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

# img = cv2.imread(images[0])
# plt.imshow(img)
# plt.title(f'Shape: {img.shape}')
# plt.show()

fig = plt.figure(figsize=(8, 3))

plt_rows = 2
plt_cols = 2
plt_iter = 1

plt.subplots_adjust(hspace=0.5)

for i in range(plt_rows*plt_cols):
    plt.subplot(plt_rows, plt_cols, plt_iter)
    
    # img_index = np.random.randint(0, len(images))
    img_index = 7
    # Load random image
    img = cv2.imread(images[img_index])
    # Covert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray,5)
    # Take binary threshold
    thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,2)
    # Invert image
    bit_not = cv2.bitwise_not(thresh)
    # Find contours
    contours, hierarchy = cv2.findContours(bit_not, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Draw on orignal image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    # Get bounding rect of each contour
    rects = [cv2.boundingRect(c) for c in contours]
    # Sort rects by their width
    rects.sort(key=lambda x: x[2])
    
    # Deal with touching letters where one wide bounding box
    # envlopes two letters. split these in half
    while len(rects) < 4:
        # Pop widest rect
        wide_rect = rects.pop()
        x, y, w, h = wide_rect
        # Split in two
        first_half = (x, y, w//2, h)
        second_half = (x+w//2, y, w//2, h)
        rects.append(first_half)
        rects.append(second_half)
        # Re-sort rects by their width
        rects.sort(key=lambda x: x[2])
    
    for rect in rects:
        x, y, w, h = rect
        # Buffer rect by 1 pixel
        cv2.rectangle(img, (x-1, y-1), (x+w+1, y+h+1), (255, 0, 0), 1)
    
    plt.imshow(img, 'gray')
    plt_iter += 1

fig = plt.figure(figsize=(8, 3))

plt_rows = 2
plt_cols = 2
plt_iter = 1

plt.subplots_adjust(hspace=0.5)

for i in range(plt_rows*plt_cols):
    plt.subplot(plt_rows, plt_cols, plt_iter)
    
    # img_index = np.random.randint(0, len(images))
    img_index = 7
    # Load random image
    img = cv2.imread(images[img_index])
    # Covert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray,5)
    # Take binary threshold
    thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,2)
    # Invert image
    bit_not = cv2.bitwise_not(thresh)
    # Find contours
    contours, hierarchy = cv2.findContours(bit_not, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Draw on orignal image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    # Get bounding rect of each contour
    rects = [cv2.boundingRect(c) for c in contours]
    # Sort rects by their width
    rects.sort(key=lambda x: x[2])
    
    # Deal with touching letters where one wide bounding box
    # envlopes two letters. split these in half
    while len(rects) < 4:
        # Pop widest rect
        wide_rect = rects.pop()
        x, y, w, h = wide_rect
        # Split in two
        first_half = (x, y, w//2, h)
        second_half = (x+w//2, y, w//2, h)
        rects.append(first_half)
        rects.append(second_half)
        # Re-sort rects by their width
        rects.sort(key=lambda x: x[2])
    
    for rect in rects:
        x, y, w, h = rect
        # Buffer rect by 1 pixel
        cv2.rectangle(img, (x-1, y-1), (x+w+1, y+h+1), (255, 0, 0), 1)
    
    plt.imshow(bit_not, 'gray')
    plt_iter += 1

