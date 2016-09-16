#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2 as cv
from colour import Color
import numpy as np
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
    '.png': 'PNG'
}


class Engine(BaseEngine):
    @property
    def image_depth(self):
        if self.image is None:
            return 8
        return self.image.dtype.itemsize

    @property
    def image_channels(self):
        if self.image is None:
            return 3
        return self.image.shape[2]

    @classmethod
    def parse_hex_color(cls, color):
        try:
            color = Color(color).get_rgb()
            return tuple(c * 255 for c in reversed(color))
        except Exception:
            return None

    def gen_image(self, size, color_value):
        img = np.zeros((size[1], size[0]), self.image.dtype, self.image_channels)
        # img0 = cv.CreateImage(size, self.image_depth, self.image_channels)
        if color_value == 'transparent':
            color = (255, 255, 255, 255)
        else:
            color = self.parse_hex_color(color_value)
            if not color:
                raise ValueError('Color %s is not valid.' % color_value)
        # cv.Set(img0, color)
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

        # imagefiledata = cv.CreateMatHeader(1, len(buffer), cv.CV_8UC1)
        # cv.SetData(imagefiledata, buffer, len(buffer))
        img = cv.imdecode(np.frombuffer(buffer, np.uint8), -1)
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
        return self.image.shape[:2]

    def normalize(self):
        pass

    def resize(self, width, height):
        r = height / self.image.shape[0]
        width = int(self.image.shape[1] * r)
        dim = (int(round(width, 0)), int(round(height, 0)))
        self.image = cv.resize(self.image, dim, interpolation=cv.INTER_AREA)

    def crop(self, left, top, right, bottom):
        self.image = self.image[top: bottom, left: right]

    def rotate(self, degrees):
        shape = self.image.shape
        image_center = tuple(np.array(shape[1], shape[0]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, degrees, 1.0)
        self.image = cv.warpAffine(self.image, rot_mat, self.image.shape, flags=cv.INTER_LINEAR)

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
                options = [cv.IMWRITE_JPEG_QUALITY, quality]
        except KeyError:
            # default is JPEG so
            options = [cv.IMWRITE_JPEG_QUALITY, quality]

        success, buf = cv.imencode(extension, self.image, options or [])
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
        # TODO: Handle other formats
        if self.image_channels == 4:
            mode = 'BGRA'
        elif self.image_channels == 3:
            mode = 'BGR'
        else:
            mode = 'BGR'
            # rgb_copy = cv.CreateImage((self.image.width, self.image.height), 8, 3)
            shape = self.image.shape
            rgb_copy = np.zeros((shape[1], shape[0]), np.uint8, 3)
            cv.cvtColor(self.image, cv.COLOR_GRAY2BGR, rgb_copy)
            self.image = rgb_copy
        return mode, self.image.tostring()

    def draw_rectangle(self, x, y, width, height):
        cv.rectangle(self.image, (int(x), int(y)), (int(x + width), int(y + height)), (255, 255, 255))

    def convert_to_grayscale(self):
        if self.image_channels >= 3:
            # FIXME: OpenCV does not support grayscale with alpha channel?
            # grayscaled = cv.CreateImage((self.image.width, self.image.height), self.image_depth, 1)
            shape = self.image.shape
            #grayscaled = np.zeros((shape[1], shape[0]), self.image.dtype, 3)
            self.image = cv.cvtColor(self.image, cv.COLOR_BGRA2GRAY)

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
            shape = self.image.shape
            print(type(shape[1]), type(shape[0]), type(self.image.dtype))
            with_alpha = np.zeros((shape[1], shape[0]), self.image.dtype, 4)
            if self.image_channels == 3:
                cv.cvtColor(self.image, cv.COLOR_BGR2BGRA, with_alpha)
            else:
                cv.CvtColor(self.image, cv.COLOR_GRAY2BGRA, with_alpha)
            self.image = with_alpha
