#!/usr/bin/env python
# -*- coding: utf-8 -*-

# thumbor imaging service - opencv engine
# https://github.com/thumbor/opencv-engine

# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license
# Copyright (c) 2014 globo.com timehome@corp.globo.com

import uuid
try:
    import cv
except ImportError:
    import cv2.cv as cv

from colour import Color
from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment
import cv2
import gdal
import numpy
from osgeo import osr

# need to monkey patch the BaseEngine.get_mimetype function to handle tiffs
# has to be patched this way b/c called as both a classmethod and instance method internally in thumbor
old_mime = BaseEngine.get_mimetype


def new_mime(buffer):
    ''' determine the mime type from the raw image data
        Args:
            buffer - raw image data
        Returns:
            mime - mime type of image
    '''
    mime = old_mime(buffer)
    # tif files start with 'II'
    if not mime and buffer.startswith('II'):
        mime = 'image/tiff'
    return mime

BaseEngine.get_mimetype = staticmethod(new_mime)

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
    '.tiff': 'TIFF',
    '.tif': 'TIFF',
}


class Engine(BaseEngine):

    @property
    def image_depth(self):
        if self.image is None:
            return 8
        return cv.GetImage(self.image).depth

    @property
    def image_channels(self):
        if self.image is None:
            return 3
        return self.image.channels

    @classmethod
    def parse_hex_color(cls, color):
        try:
            color = Color(color).get_rgb()
            return tuple(c * 255 for c in reversed(color))
        except Exception:
            return None

    def gen_image(self, size, color_value):
        img0 = cv.CreateImage(size, self.image_depth, self.image_channels)
        if color_value == 'transparent':
            color = (255, 255, 255, 255)
        else:
            color = self.parse_hex_color(color_value)
            if not color:
                raise ValueError('Color %s is not valid.' % color_value)
        cv.Set(img0, color)
        return img0

    def read(self, extension=None, quality=None):
        if not extension and FORMATS[self.extension] == 'TIFF':
            # If the image loaded was a tiff, return the buffer created earlier.
            return self.buffer
        else:
            if quality is None:
                quality = self.context.config.QUALITY
            options = None
            self.extension = extension or self.extension

            # Check if we should write a JPEG. If we are allowing defaulting to jpeg
            # and if the alpha channel is all white (opaque).
            channels = None
            if hasattr(self.context, 'request') and getattr(self.context.request, 'default_to_jpeg', True):
                channels = cv2.split(numpy.asarray(self.image))
                if len(channels) > 3 and numpy.all(channels[3] == 255):
                    self.extension = '.jpg'

            try:
                if FORMATS[self.extension] == 'JPEG':
                    options = [cv.CV_IMWRITE_JPEG_QUALITY, quality]
            except KeyError:
                # default is JPEG so
                options = [cv.CV_IMWRITE_JPEG_QUALITY, quality]

            if FORMATS[self.extension] == 'TIFF':
                channels = channels or cv2.split(numpy.asarray(self.image))
                data = self.write_channels_to_tiff_buffer(channels)
            else:
                data = cv.EncodeImage(self.extension, self.image, options or []).tostring()

            if FORMATS[self.extension] == 'JPEG' and self.context.config.PRESERVE_EXIF_INFO:
                if hasattr(self, 'exif'):
                    img = JpegFile.fromString(data)
                    img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                    data = img.writeString()

        return data

    def create_image(self, buffer, create_alpha=True):
        self.extension = self.extension or '.tif'
        self.no_data_value = None
        # FIXME: opencv doesn't support gifs, even worse, the library
        # segfaults when trying to decoding a gif. An exception is a
        # less drastic measure.
        try:
            if FORMATS[self.extension] == 'GIF':
                raise ValueError("opencv doesn't support gifs")
        except KeyError:
            pass

        if FORMATS[self.extension] == 'TIFF':
            self.buffer = buffer
            img0 = self.read_tiff(buffer, create_alpha)
        else:
            imagefiledata = cv.CreateMatHeader(1, len(buffer), cv.CV_8UC1)
            cv.SetData(imagefiledata, buffer, len(buffer))
            img0 = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_UNCHANGED)

        if FORMATS[self.extension] == 'JPEG':
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass
        return img0

    def read_tiff(self, buffer, create_alpha=True):
        """ Reads image using GDAL from a buffer, and returns a CV2 image.
        """
        mem_map_name = '/vsimem/{}'.format(uuid.uuid4().get_hex())
        gdal_img = None
        try:
            gdal.FileFromMemBuffer(mem_map_name, buffer)
            gdal_img = gdal.Open(mem_map_name)

            channels = [gdal_img.GetRasterBand(i).ReadAsArray() for i in range(1, gdal_img.RasterCount + 1)]

            if len(channels) >= 3:  # opencv is bgr not rgb.
                red_channel = channels[0]
                channels[0] = channels[2]
                channels[2] = red_channel

            if len(channels) < 4 and create_alpha:
                self.no_data_value = gdal_img.GetRasterBand(1).GetNoDataValue()
                channels.append(numpy.float32(gdal_img.GetRasterBand(1).GetMaskBand().ReadAsArray()))
            return cv.fromarray(cv2.merge(channels))
        finally:
            gdal_img = None
            gdal.Unlink(mem_map_name)  # Cleanup.

    def read_vsimem(self, fn):
        """Read GDAL vsimem files"""
        vsifile = None
        try:
            vsifile = gdal.VSIFOpenL(fn, 'r')
            gdal.VSIFSeekL(vsifile, 0, 2)
            vsileng = gdal.VSIFTellL(vsifile)
            gdal.VSIFSeekL(vsifile, 0, 0)
            return gdal.VSIFReadL(1, vsileng, vsifile)
        finally:
            if vsifile:
                gdal.VSIFCloseL(vsifile)

    def write_channels_to_tiff_buffer(self, channels):
        mem_map_name = '/vsimem/{}.tiff'.format(uuid.uuid4().get_hex())
        driver = gdal.GetDriverByName('GTiff')
        w, h = channels[0].shape
        gdal_img = None
        try:
            if len(channels) == 1:
                # DEM Tiff (32 bit floating point single channel)
                gdal_img = driver.Create(mem_map_name, w, h, len(channels), gdal.GDT_Float32)
                outband = gdal_img.GetRasterBand(1)
                outband.WriteArray(channels[0], 0, 0)
                outband.SetNoDataValue(-32767)
                outband.FlushCache()
                outband = None
                gdal_img.FlushCache()

                self.set_geo_info(gdal_img)
                return self.read_vsimem(mem_map_name)
            elif len(channels) == 4:
                # BGRA 8 bit unsigned int.
                gdal_img = driver.Create(mem_map_name, h, w, len(channels), gdal.GDT_Byte)
                band_order = [2, 1, 0, 3]
                img_bands = [gdal_img.GetRasterBand(i) for i in range(1, 5)]
                for outband, band_i in zip(img_bands, band_order):
                    outband.WriteArray(channels[band_i], 0, 0)
                    outband.SetNoDataValue(-32767)
                    outband.FlushCache()
                    del outband
                del img_bands

                self.set_geo_info(gdal_img)
                return self.read_vsimem(mem_map_name)
        finally:
            del gdal_img
            gdal.Unlink(mem_map_name)  # Cleanup.

    def set_geo_info(self, gdal_img):
        """ Set the georeferencing information for the given gdal image.
        """
        if hasattr(self.context.request, 'geo_info'):
            geo = self.context.request.geo_info
            gdal_img.SetGeoTransform([geo['upper_left_x'], geo['resx'], 0, geo['upper_left_y'], 0, -geo['resy']])

        # Set projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        gdal_img.SetProjection(srs.ExportToWkt())
        gdal_img.FlushCache()
        del srs

    @property
    def size(self):
        return cv.GetSize(self.image)

    def normalize(self):
        pass

    def resize(self, width, height):
        thumbnail = cv.CreateImage(
            (int(round(width, 0)), int(round(height, 0))),
            self.image_depth,
            self.image_channels
        )
        cv.Resize(self.image, thumbnail, cv.CV_INTER_AREA)
        self.image = thumbnail

    def crop(self, left, top, right, bottom):
        new_width = right - left
        new_height = bottom - top
        cropped = cv.CreateImage(
            (new_width, new_height), self.image_depth, self.image_channels
        )
        src_region = cv.GetSubRect(self.image, (left, top, new_width, new_height))
        cv.Copy(src_region, cropped)

        self.image = cropped

    def rotate(self, degrees):
        if (degrees > 180):
            # Flip around both axes
            cv.Flip(self.image, None, -1)
            degrees = degrees - 180

        img = self.image
        size = cv.GetSize(img)

        if (degrees / 90 % 2):
            new_size = (size[1], size[0])
            center = ((size[0] - 1) * 0.5, (size[0] - 1) * 0.5)
        else:
            new_size = size
            center = ((size[0] - 1) * 0.5, (size[1] - 1) * 0.5)

        mapMatrix = cv.CreateMat(2, 3, cv.CV_64F)
        cv.GetRotationMatrix2D(center, degrees, 1.0, mapMatrix)
        dst = cv.CreateImage(new_size, self.image_depth, self.image_channels)
        cv.SetZero(dst)
        cv.WarpAffine(img, dst, mapMatrix)
        self.image = dst

    def flip_vertically(self):
        cv.Flip(self.image, None, 1)

    def flip_horizontally(self):
        cv.Flip(self.image, None, 0)

    def set_image_data(self, data):
        cv.SetData(self.image, data)

    def image_data_as_rgb(self, update_image=True):
        # TODO: Handle other formats
        if self.image_channels == 4:
            mode = 'BGRA'
        elif self.image_channels == 3:
            mode = 'BGR'
        else:
            mode = 'BGR'
            rgb_copy = cv.CreateImage((self.image.width, self.image.height), 8, 3)
            cv.CvtColor(self.image, rgb_copy, cv.CV_GRAY2BGR)
            self.image = rgb_copy
        return mode, self.image.tostring()

    def draw_rectangle(self, x, y, width, height):
        cv.Rectangle(self.image, (int(x), int(y)), (int(x + width), int(y + height)), cv.Scalar(255, 255, 255, 1.0))

    def convert_to_grayscale(self):
        if self.image_channels >= 3:
            # FIXME: OpenCV does not support grayscale with alpha channel?
            grayscaled = cv.CreateImage((self.image.width, self.image.height), self.image_depth, 1)
            cv.CvtColor(self.image, grayscaled, cv.CV_BGRA2GRAY)
            self.image = grayscaled

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
            with_alpha = cv.CreateImage(
                (self.image.width, self.image.height), self.image_depth, 4
            )
            if self.image_channels == 3:
                cv.CvtColor(self.image, with_alpha, cv.CV_BGR2BGRA)
            else:
                cv.CvtColor(self.image, with_alpha, cv.CV_GRAY2BGRA)
            self.image = with_alpha
