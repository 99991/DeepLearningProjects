import scipy as sp
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

image = scipy.misc.imread("lena.bmp")

image = scipy.misc.imrotate(image, 20)

plt.imshow(image)
plt.show()
