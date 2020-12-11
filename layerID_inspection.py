"""A script to plot an image in a pyplot figure

    Hover mouse over pyplot figure to determine the max/min
    values of x and y that totally enclose the flake.

    Enter those values for the indices of the plotted image,
    then run script again to determine regions of substrate.
    """

import matplotlib.pyplot as plt
import cv2

img_loc = '.\\Flake Images\\Graphene\\RSGR003\\All\\4A1.jpg' ## Sample flake image location
img = cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)
print(img.shape)
plt.imshow(cv2.cvtColor(img[500:1300,1000:1600], cv2.COLOR_BGR2RGB)) ## crop image with indices

plt.show()
