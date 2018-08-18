#!/usr/bin/env python
import numpy as np
import PIL
from PIL import Image

class ImageVector(object):
    """vectorize images
    Attributes:
    """
    def __init__(self):
        super(ImageVector, self).__init__()

    def vectorize(self, image, image_size):
        """vectorize images
        Args:
            image           object of PIL.Image.Image
            image_size      (channel, height, width)
        Returns:
            img_vec         numpy.array
        Raises:
            TypeError       
        """
        if not isinstance(image, PIL.Image.Image):
            raise TypeError("image must be object of PIL.Image.Image")

        img_vec = np.asarray(image, dtype='float64')
        img_vec = img_vec.transpose(2,0,1)
        img_vec = img_vec.reshape((np.prod(image_size),))

        return img_vec
