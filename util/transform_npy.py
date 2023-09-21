import os
import numpy as np
from scipy.ndimage import rotate
import torch
from torchvision import transforms

import cv2
from torchvision import transforms

# Define the class for resizing with padding
class AddingPad:
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, data):
        h, w = data.shape
        
        new_h, new_w = self.output_size
        top = (new_h - h) // 2
        bottom = new_h - h - top
        left = (new_w - w) // 2
        right = new_w - w - left
        
        padded_data = np.pad(data, ((top, bottom), (left, right)), mode='constant')
        return padded_data
    


class ResizeNumpy:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, np_array):
        h, w = np_array.shape[:2]
        new_h, new_w = self.output_size

        # Redimensionar o array NumPy usando a interpolação bilinear
        resized_array = cv2.resize(np_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return resized_array



# Define the class for random horizontal flipping
class RandomHorizontalFlipNpy:
    def __call__(self, data):
        if np.random.rand() < 0.5:
            data = np.fliplr(data)
        return data

# Define the class for random rotation
class RandomRotationNpy:
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, data):
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        rotated_data = rotate(data, angle, reshape=False, mode='constant', cval=0.0)
        return rotated_data