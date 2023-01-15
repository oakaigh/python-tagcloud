from __future__ import annotations


from . import backend_base
from . import graphics

import math
import collections

import numpy as np
import numpy.typing

from . import utils

import PIL
import PIL.Image
from PIL.Image import (
    Resampling,
    Transpose,
    Transform
)
import PIL.ImageDraw
import PIL.ImageFont


class TaskQueue:
    def __init__(self):
        self._base = collections.deque()

    def get(self):
        return self._base.popleft()

    def put(self, o):
        return self._base.append(o)

    def empty(self):
        return len(self._base) == 0

class Image:
    @classmethod
    def make(cls, **options):
        return cls(
            PIL.Image.new(**options)
        )
        
    @classmethod
    def from_internal(cls, o):
        return cls(
            PIL.Image.Image()._new(o)
        )

    @classmethod
    def from_font(
        cls, 
        font: PIL.ImageFont.ImageFont,
        **options
    ):
        return cls.from_internal(
            font.getmask(**options)
        )

    @property
    def _base(self) -> PIL.Image.Image:
        return self.__base

    @_base.setter
    def _base(self, v: PIL.Image.Image) -> PIL.Image.Image:
        self.__base = v
        # invalidate context
        self._context = None
        return self.__base

    @property
    def _context(self) -> PIL.ImageDraw.Draw:
        if self.__context is None:
            self.__context = PIL.ImageDraw.Draw(self._base)
        return self.__context

    @_context.setter
    def _context(self, v: PIL.ImageDraw.Draw) -> PIL.ImageDraw.Draw:
        self.__context = v
        return self.__context

    @property
    def _dimension(self) -> graphics.Dimension:
        try:
            self.__dimension
        except AttributeError:
            self.__dimension = graphics.Dimension.make(
                width=self._base.width, 
                height=self._base.height
            )
        return self.__dimension

    @_dimension.setter
    def _dimension(self, v: graphics.Dimension) -> graphics.Dimension:
        self.__dimension = v
        return self.__dimension

    def __init__(
        self, 
        base: PIL.Image.Image, 
        **options
    ):
        self._base = base
        self.set(**options)

    def set(
        self, 
        nocopy=False, 
        norender=False
    ) -> Image:
        self._nocopy = nocopy
        self._render_queue = (
            None if norender else 
            getattr(self, '_render_queue', None) or TaskQueue()
        )
        return self

    def render(self) -> Image:
        if self._render_queue is None:
            return self

        while not self._render_queue.empty():
            f = self._render_queue.get()
            f.__call__()

        return self

    @property
    def size(self) -> graphics.Dimension:
        return self._dimension

    @property
    def data(self) -> np.typing.NDArray:
        return np.asarray(self._base)

    def copy(self, *args, **kwargs) -> Image:
        if self._nocopy:
            return self

        return self.__class__(
            self._base.copy(*args, **kwargs)
        )
        
    def transpose(
        self, 
        method: Transpose, 
        *args, **kwargs
    ):
        if method is None:
            return

        def f():
            self._base = self._base.transpose(method, *args, **kwargs)
        self._render_queue.put(f)            

        if method in (None, Transpose.ROTATE_180):
            pass

        if method in (
            Transpose.ROTATE_90,
            Transpose.ROTATE_270
        ):
            self._dimension = self._dimension.transpose()
        
    def transform(
        self,
        size: graphics.Dimension,
        *args, **kwargs
    ):
        def f():
            self._base = self._base.transform(size, *args, **kwargs)
        self._render_queue.put(f)

        self._dimension = size

    # stolen from pillow
    # see https://github.com/python-pillow/Pillow/blob/9.4.x/src/PIL/Image.py#L2227-L2344
    def rotate(
        self,
        angle,
        resample=Resampling.NEAREST,
        expand=0,
        center=None,
        translate=None,
        fillcolor=None,
    ):
        """
        Returns a rotated copy of this image.  This method returns a
        copy of this image, rotated the given number of degrees counter
        clockwise around its centre.

        :param angle: In degrees counter clockwise.
        :param resample: An optional resampling filter.  This can be
           one of :py:data:`Resampling.NEAREST` (use nearest neighbour),
           :py:data:`Resampling.BILINEAR` (linear interpolation in a 2x2
           environment), or :py:data:`Resampling.BICUBIC` (cubic spline
           interpolation in a 4x4 environment). If omitted, or if the image has
           mode "1" or "P", it is set to :py:data:`Resampling.NEAREST`.
           See :ref:`concept-filters`.
        :param expand: Optional expansion flag.  If true, expands the output
           image to make it large enough to hold the entire rotated image.
           If false or omitted, make the output image the same size as the
           input image.  Note that the expand flag assumes rotation around
           the center and no translation.
        :param center: Optional center of rotation (a 2-tuple).  Origin is
           the upper left corner.  Default is the center of the image.
        :param translate: An optional post-rotate translation (a 2-tuple).
        :param fillcolor: An optional color for area outside the rotated image.
        :returns: An :py:class:`~PIL.Image.Image` object.
        """

        angle = angle % 360.0

        # Fast paths regardless of filter, as long as we're not
        # translating or changing the center.
        if not (center or translate):
            if angle == 0:
                return self.transpose(None)
            if angle == 180:
                return self.transpose(Transpose.ROTATE_180)
            if angle in (90, 270) and (expand or self.size.width == self.size.height):
                return self.transpose(
                    Transpose.ROTATE_90 if angle == 90 else Transpose.ROTATE_270
                )

        # Calculate the affine matrix.  Note that this is the reverse
        # transformation (from destination image to source) because we
        # want to interpolate the (discrete) destination pixel from
        # the local area around the (floating) source pixel.

        # The matrix we actually want (note that it operates from the right):
        # (1, 0, tx)   (1, 0, cx)   ( cos a, sin a, 0)   (1, 0, -cx)
        # (0, 1, ty) * (0, 1, cy) * (-sin a, cos a, 0) * (0, 1, -cy)
        # (0, 0,  1)   (0, 0,  1)   (     0,     0, 1)   (0, 0,   1)

        # The reverse matrix is thus:
        # (1, 0, cx)   ( cos -a, sin -a, 0)   (1, 0, -cx)   (1, 0, -tx)
        # (0, 1, cy) * (-sin -a, cos -a, 0) * (0, 1, -cy) * (0, 1, -ty)
        # (0, 0,  1)   (      0,      0, 1)   (0, 0,   1)   (0, 0,   1)

        # In any case, the final translation may be updated at the end to
        # compensate for the expand flag.

        w, h = self.size

        if translate is None:
            post_trans = (0, 0)
        else:
            post_trans = translate
        if center is None:
            # FIXME These should be rounded to ints?
            rotn_center = (w / 2.0, h / 2.0)
        else:
            rotn_center = center

        angle = -math.radians(angle)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]

        if expand:
            # calculate output size
            xx = []
            yy = []
            for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
                x, y = transform(x, y, matrix)
                xx.append(x)
                yy.append(y)
            nw = math.ceil(max(xx)) - math.floor(min(xx))
            nh = math.ceil(max(yy)) - math.floor(min(yy))

            # We multiply a translation matrix from the right.  Because of its
            # special form, this is the same as taking the image of the
            # translation vector as new translation vector.
            matrix[2], matrix[5] = transform(-(nw - w) / 2.0, -(nh - h) / 2.0, matrix)
            w, h = nw, nh

        return self.transform(
            graphics.Dimension(w, h), Transform.AFFINE, matrix, resample, fillcolor=fillcolor
        )

    def crop(self, box: graphics.BBox=None):
        if box is None:
            return

        def f():
            self._base = self._base.crop(box=box)
        self._render_queue.put(f)

        self._dimension = box.dimension()

    def paste(self, image: Image, box: graphics.BBox=None, mask: Image=None):
        def f():
            self._base.paste(image._base, box=box, mask=mask._base)
        self._render_queue.put(f)

    def text(self, expand=False, **options) -> graphics.BBox:
        bbox = graphics.BBox(*self._context.textbbox(**utils.subdict(
            options, 
            keys=[
                'xy',
                'text',
                'font',
                'anchor',
                'spacing',
                'align',
                'direction',
                'features',
                'language',
                'stroke_width',
                'embedded_color'
            ],
            ignore_nonexist=True
        )))

        if expand and bbox not in graphics.BBox.from_dimension(self.size):
            self.crop(box=graphics.BBox(
                left=0, top=0, 
                right=bbox.right, bottom=bbox.bottom
            ))
        
        def f():
            self._context.text(**options)
        self._render_queue.put(f)

        return bbox

# TODO
class Font:
    pass



import io

class CanvasPIL(backend_base.CanvasBase):
    def __init__(self, size: graphics.Dimension):
        super().__init__()
        # TODO
        self._base = Image.make(mode='1', size=size.transpose(), color=0)
        #self._base = Image.make(mode='L', size=size.transpose(), color=0)
        # TODO !!!!!!!
        with open('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', mode='rb') as f:
            self._font_data = f.read()

    @property
    def dimension(self) -> graphics.Dimension:
        return self._base.size.transpose()

    @property
    def data_bilevel(self) -> graphics.BilevelData:
        return self._base.data

    def text(self, text_spec: backend_base.TextSpec) -> graphics.Dimension:
        # TODO font manager!!! 
        font = PIL.ImageFont.truetype(
            io.BytesIO(self._font_data), 
            size=int(text_spec.size)
        )

        # TODO memorization
        region = Image.from_font(font, mode='1', text=text_spec.content)

        # TODO memorization
        region.rotate(angle=int(text_spec.rotation), expand=True)

        if text_spec.position is not None:
            pos = text_spec.position

            region.render()
            self._base.paste(
                region, 
                # NOTE pillow uses column major `pos` is row major
                box=pos.transpose(), 
                mask=region
            )
            self._base.render()
            # TODO
            self.callbacks.region_update.__call__(
                self, 
                pos, 
                region.data
            )

        return region.size.transpose()
