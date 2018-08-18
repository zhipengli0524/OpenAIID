#!/usr/bin/env python
# -*- coding:utf8 -*-

import PIL
from PIL import Image
import sys
import logging
import os

class ImageCutter(object):
    """cut images
    Attributes:
    """
    def __init__(self):
        super(ImageCutter, self).__init__()

    def cut_by_half_center(self, image, resize=None):
        """Cut the Image by hale center
        Args:
            image   str (path of image file) or Image object
        Returns:
            PIL.Image.Image
        Raises:
            TypeError
        """
        im = self._openImage(image)
        x_size, y_size = im.size
        start_point_xy = x_size / 4
        end_point_xy = x_size / 4 + x_size / 2
        box = (start_point_xy, start_point_xy, 
               end_point_xy, end_point_xy)
        new_im = im.crop(box)
        if resize is not None:
            new_im = self.resize(new_im, resize)
        return new_im

    def resize(self, image, size):
        """Resize the input image
        Args:
            image   PIL.Image.Image object
            size    size of output
        Returns:
            object  PIL.Image.Image
        Raises:
            TypeError
        """
        if not isinstance(image, PIL.Image.Image):
            raise TypeError("image should be object of PIL.Image.Image")
        if (not isinstance(size, tuple)) or len(size) < 2:
            raise TypeError("size shoule be tuple of which length is at least 2")

        new_im = image.resize(size[:2])
        return new_im

    def _openImage(self, image):
        """To get object of PIL.Image.Image
        Args:
            image  str (path of image file) or Image object 
        Returns:
            PIL.Image.Image
        Raises:
            TypeError
        """
        if isinstance(image, str):
            logging.info("loading image from file: %s" % image)
            im = Image.open(image)
        elif isinstance(image, PIL.Image.Image):
            im = image
        else:
            raise TypeError("Input Type Error: only str (path of image file) \
                             and PIL.Image.Image object acceptable.")
        return im



