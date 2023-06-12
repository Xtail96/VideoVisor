import cv2
import skimage
import skimage.filters
import skimage.restoration
import scipy
import scipy.ndimage
import scipy.signal
import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib.pyplot import figure
#from numba import njit, jit
from math import *


# Общая для всех зашумлённость
#amount = 0.05
amount = 0.025
#var = 0.1
var = 0.01
mean = 0.0
#lam = 12.0
lam = 0.01


def add_impulse_noise(image):
    # Наложение импульсного шума
    image_noised_s_and_p = skimage.util.random_noise(image, mode="s&p", amount=amount, salt_vs_pepper=0.3)
    image_noised_s_and_p = np.clip(image_noised_s_and_p, 0.0, 1.0)
    image_noised_s_and_p = skimage.img_as_ubyte(image_noised_s_and_p)
    return image_noised_s_and_p


def add_multiplicative_noise(image):
    # Наложение мультипликативного шума
    image_noised_multi = skimage.util.random_noise(image, mode="speckle", var=var, mean=mean)
    image_noised_multi = np.clip(image_noised_multi, 0.0, 1.0)
    image_noised_multi = skimage.img_as_ubyte(image_noised_multi)
    return image_noised_multi


def add_gaussian_noise(image):
    # Наложение Гауссова шума
    image_noised_gaussian = skimage.util.random_noise(image, mode="gaussian", var=var, mean=mean)
    image_noised_gaussian = np.clip(image_noised_gaussian, 0.0, 1.0)
    image_noised_gaussian = skimage.img_as_ubyte(image_noised_gaussian)
    return image_noised_gaussian


def add_quantization_noise(image):
    # Создание шума квантования с заданными параметрами распределения Пуассона
    noise_poisson = np.random.poisson(lam=lam, size=image.shape) * (1.0 / 255.0)
    noise_poisson = np.clip(noise_poisson, 1.0 / 255.0, 1.0)

    # Наложение шума квантования
    image_noised_poisson = skimage.util.random_noise(image, mode="localvar", local_vars=noise_poisson)
    image_noised_poisson = np.clip(image_noised_poisson, 0.0, 1.0)
    image_noised_poisson = skimage.img_as_ubyte(image_noised_poisson)
    return image_noised_poisson.copy()


def add_noise(image_path: str):
    image = cv2.imread(image_path)
    image = add_impulse_noise(image)
    image = add_multiplicative_noise(image)
    image = add_gaussian_noise(image)
    image = add_quantization_noise(image)
    cv2.imwrite(image_path, image)



