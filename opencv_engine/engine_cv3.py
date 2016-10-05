#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license
# Copyright (c) 2016 fanhero.com christian@fanhero.com

import cv2
import numpy as np

from colour import Color
from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment

try:
    from thumbor.ext.filters import _composite
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False

FORMATS = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
    '.png': 'PNG',
    '.webp': 'WEBP'
}


class Engine(BaseEngine):
    @property
    def image_depth(self):
        if self.image is None:
            return np.uint8
        return self.image.dtype

    @property
    def image_channels(self):
        if self.image is None:
            return 3
        # if the image is grayscale
        try:
            return self.image.shape[2]
        except IndexError:
            return 1

    @classmethod
    def parse_hex_color(cls, color):
        try:
            color = Color(color).get_rgb()
            return tuple(c * 255 for c in reversed(color))
        except Exception:
            return None

    def gen_image(self, size, color_value):
        if color_value == 'transparent':
            color = (255, 255, 255, 255)
            img = np.zeros((size[1], size[0], 4), self.image_depth)
        else:
            img = np.zeros((size[1], size[0], self.image_channels), self.image_depth)
            color = self.parse_hex_color(color_value)
            if not color:
                raise ValueError('Color %s is not valid.' % color_value)
        img[:] = color
        return img

    def create_image(self, buffer):
        # FIXME: opencv doesn't support gifs, even worse, the library
        # segfaults when trying to decoding a gif. An exception is a
        # less drastic measure.
        try:
            if FORMATS[self.extension] == 'GIF':
                raise ValueError("opencv doesn't support gifs")
        except KeyError:
            pass

        img = cv2.imdecode(np.frombuffer(buffer, np.uint8), -1)
        if FORMATS[self.extension] == 'JPEG':
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass
        return img

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]

    def normalize(self):
        pass

    def resize(self, width, height):
        r = height / self.size[1]
        width = int(self.size[0] * r)
        dim = (int(round(width, 0)), int(round(height, 0)))
        self.image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)

    def crop(self, left, top, right, bottom):
        self.image = self.image[top: bottom, left: right]

    def rotate(self, degrees):
        image_center = (self.size[1] / 2, self.size[0] / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, degrees, 1.0)
        self.image = cv2.warpAffine(self.image, rot_mat, dsize=self.size)

    def flip_vertically(self):
        self.image = np.flipud(self.image)

    def flip_horizontally(self):
        self.image = np.fliplr(self.image)

    def read(self, extension=None, quality=None):
        if quality is None:
            quality = self.context.config.QUALITY

        options = None
        extension = extension or self.extension
        try:
            if FORMATS[extension] == 'JPEG':
                options = [cv2.IMWRITE_JPEG_QUALITY, quality]
        except KeyError:
            # default is JPEG so
            options = [cv2.IMWRITE_JPEG_QUALITY, quality]

        try:
            if FORMATS[extension] == 'WEBP':
                options = [cv2.IMWRITE_WEBP_QUALITY, quality]
        except KeyError:
            options = [cv2.IMWRITE_JPEG_QUALITY, quality]

        success, buf = cv2.imencode(extension, self.image, options or [])
        data = buf.tostring()

        if FORMATS[extension] == 'JPEG' and self.context.config.PRESERVE_EXIF_INFO:
            if hasattr(self, 'exif'):
                img = JpegFile.fromString(data)
                img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                data = img.writeString()

        return data

    def set_image_data(self, data):
        self.image = np.frombuffer(data, dtype=self.image.dtype).reshape(self.image.shape)

    def image_data_as_rgb(self, update_image=True):
        if self.image_channels == 4:
            mode = 'BGRA'
        elif self.image_channels == 3:
            mode = 'BGR'
        else:
            mode = 'BGR'
            rgb_copy = np.zeros((self.size[1], self.size[0], 3), self.image.dtype)
            cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR, rgb_copy)
            self.image = rgb_copy
        return mode, self.image.tostring()

    def draw_rectangle(self, x, y, width, height):
        cv2.rectangle(self.image, (int(x), int(y)), (int(x + width), int(y + height)), (255, 255, 255))

    def convert_to_grayscale(self, update_image=True, with_alpha=True):
        if self.image_channels is 4 and with_alpha:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        elif self.image_channels > 1:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            #image = self.image
            #Already grayscale, no need to filter
        else:
            image = self.image
        if update_image:
            self.image = image.astype(np.uint8)
        return image.astype(np.uint8)

    def paste(self, other_engine, pos, merge=True):
        if merge and not FILTERS_AVAILABLE:
            raise RuntimeError(
                'You need filters enabled to use paste with merge. Please reinstall ' +
                'thumbor with proper compilation of its filters.')

        self.enable_alpha()
        other_engine.enable_alpha()

        sz = self.size
        other_size = other_engine.size

        mode, data = self.image_data_as_rgb()
        other_mode, other_data = other_engine.image_data_as_rgb()

        imgdata = _composite.apply(
            mode, data, sz[0], sz[1],
            other_data, other_size[0], other_size[1], pos[0], pos[1], merge)

        self.set_image_data(imgdata)

    def enable_alpha(self):
        if self.image_channels < 4:
            with_alpha = np.zeros((self.size[1], self.size[0], 4), self.image.dtype)
            if self.image_channels == 3:
                cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA, with_alpha)
            else:
                cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGRA, with_alpha)
            self.image = with_alpha
