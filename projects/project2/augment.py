import scipy as sp
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import time

def rand(a, b):
    return a + (b - a)*np.random.rand()

def get_rotated_image(image, max_angle):
    angle = rand(-max_angle, max_angle)
    return scipy.misc.imrotate(image, angle)

def get_flipped_image(image):
    return image if np.random.rand() < 0.5 else np.fliplr(image)

def get_scaled_cropped_image(image, max_scale = 2.0):
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

def get_random_brightness(image, a, b, c, d):
    image = image.astype(np.float32)/256.0
    image = image*rand(a, b) + rand(c, d)
    image = np.clip(image, 0.0, 1.0)
    return image

def get_augmented_image(image):
    image = get_flipped_image(image)
    image = get_scaled_cropped_image(image, 1.5)
    image = get_rotated_image(image, 10)
    image = get_random_brightness(image, 0.9, 1.3, -0.1, 0.2)
    return image

if __name__ == '__main__':
    image = scipy.misc.imread("lena.bmp")

    for i in range(1, 4*4+1):
        plt.subplot(4, 4, i)
        plt.axis('off')
        plt.imshow(get_augmented_image(image))
    plt.show()
