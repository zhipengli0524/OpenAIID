#!/usr/bin/env python
# -*- coding:utf8 -*-

import PIL
from PIL import Image
import sys
import logging
import os
import numpy
import cv2


class ImageCutter(object):
    """cut images
    Attributes:
    """
    def __init__(self, cv_model_file=None):
        super(ImageCutter, self).__init__()
        if cv_model_file is not None:
            self._cv_model_file = cv_model_file
        else:
            file_dir = os.path.dirname(__file__)
            file_dir = os.path.abspath(file_dir)
            self._cv_model_file = os.path.join(file_dir, "cv_model", "frontalface.xml")
        self._face_cascade = cv2.CascadeClassifier(self._cv_model_file)

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

    def cut_by_opencv(self, image, resize=None):
        """Cut the image by open cv
        Args:
            image str (path of image file) or Image object
        Returns:
            PIL.Image.Image
        Raises:
            TypeError
        """
        im = self._openImage(image)
        imcv = cv2.cvtColor(numpy.asarray(im), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(imcv, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
                   gray,
                   scaleFactor = 1.2, 
                   minNeighbors = 5,
                   minSize = (5,5),
                   flags = 0
                )
        if len(faces) <= 0:
            return None

        face = faces[self._bigIndex(faces)]
        x, y, w, h = face
        new_imcv = imcv[y:y+h, x:x+w]
        new_im = Image.fromarray(cv2.cvtColor(new_imcv, cv2.COLOR_BGR2RGB))

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

    def _bigIndex(self, faces):
        """To get index of biggest face
        Args:
            faces   list [(x, y, w, h), (...), ...]
        Returns:
            int     index
        Raises:
            None
        """
        if len(faces) <= 0:
            return -1
        
        bigIndex = 0
        aera = 0
        for idx in range(len(faces)):
            x, y, w, h = faces[idx]
            thisAera = w * h
            if thisAera > aera:
                aera = thisAera
                bigIndex = idx

        return bigIndex



