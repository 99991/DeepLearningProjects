import scipy as sp
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import time

image = scipy.misc.imread("lena.bmp")

max_angle = 30
max_scale = 2.0

def rand(a, b):
    return a + (b - a)*np.random.rand()

def get_rotated_image(image):
    angle = rand(-max_angle, max_angle)
    return scipy.misc.imrotate(image, angle)

def get_flipped_image(image):
    return image if np.random.rand() < 0.5 else np.fliplr(image)

def get_scaled_cropped_image(image):
    ny, nx, channels = image.shape
    scale = rand(1, max_scale)
    mx = int(nx*scale)
    my = int(ny*scale)
    image = scipy.misc.imresize(image, (my, mx, channels))

    if mx > nx and my > ny:
        x0 = np.random.randint(mx - nx)
        y0 = np.random.randint(my - ny)
        x1 = x0 + nx
        y1 = y0 + ny
        image = image[y0:y1, x0:x1, :]

    return image

def get_random_brightness(image):
    image = image.astype(np.float32)/256.0
    image = image*rand(0.8, 2.0) + rand(-0.2, 0.3)
    image = np.clip(image, 0.0, 1.0)
    return image

def get_modified_image(image):
    image = get_flipped_image(image)
    image = get_scaled_cropped_image(image)
    image = get_rotated_image(image)
    image = get_random_brightness(image)
    return image

for i in range(1, 4*4+1):
    plt.subplot(4, 4, i)
    plt.axis('off')
    plt.imshow(get_modified_image(image))
plt.show()
