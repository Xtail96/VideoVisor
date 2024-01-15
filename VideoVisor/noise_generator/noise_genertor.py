import cv2
import skimage
import skimage.filters
import skimage.restoration
import numpy as np


class NoiseGenerator:
    def __init__(self, *, amount=0.05, var=0.1, mean=0.0, lam=12.0):
        self.amount = amount
        self.var = var
        self.mean = mean
        self.lam = lam

    def add_impulse_noise(self, image):
        # Наложение импульсного шума.
        image_noised_s_and_p = skimage.util.random_noise(image, mode="s&p", amount=self.amount, salt_vs_pepper=0.3)
        image_noised_s_and_p = np.clip(image_noised_s_and_p, 0.0, 1.0)
        image_noised_s_and_p = skimage.img_as_ubyte(image_noised_s_and_p)
        return image_noised_s_and_p

    def add_multiplicative_noise(self, image):
        # Наложение мультипликативного шума.
        image_noised_multi = skimage.util.random_noise(image, mode="speckle", var=self.var, mean=self.mean)
        image_noised_multi = np.clip(image_noised_multi, 0.0, 1.0)
        image_noised_multi = skimage.img_as_ubyte(image_noised_multi)
        return image_noised_multi

    def add_gaussian_noise(self, image):
        # Наложение Гауссова шума.
        image_noised_gaussian = skimage.util.random_noise(image, mode="gaussian", var=self.var, mean=self.mean)
        image_noised_gaussian = np.clip(image_noised_gaussian, 0.0, 1.0)
        image_noised_gaussian = skimage.img_as_ubyte(image_noised_gaussian)
        return image_noised_gaussian

    def add_quantization_noise(self, image):
        # Создание шума квантования с заданными параметрами распределения Пуассона.
        poisson_noise = np.random.poisson(lam=self.lam, size=image.shape) * (1.0 / 255.0)
        poisson_noise = np.clip(poisson_noise, 1.0 / 255.0, 1.0)

        # Наложение шума квантования.
        image_noised_poisson = skimage.util.random_noise(image, mode="localvar", local_vars=poisson_noise)
        image_noised_poisson = np.clip(image_noised_poisson, 0.0, 1.0)
        image_noised_poisson = skimage.img_as_ubyte(image_noised_poisson)
        return image_noised_poisson.copy()

    def add_noise(self, image_path: str):
        # Наложение всех доступных типов шумов на изображение.
        image = cv2.imread(image_path)
        image = self.add_impulse_noise(image)
        image = self.add_multiplicative_noise(image)
        image = self.add_gaussian_noise(image)
        image = self.add_quantization_noise(image)
        cv2.imwrite(image_path, image)
