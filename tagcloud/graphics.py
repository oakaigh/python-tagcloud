from __future__ import annotations


import typing
import functools
import enum

import numpy as np
import numpy.typing

import numba


# see https://stackoverflow.com/a/27087513/11934495
def _TODO_shift_full(a, offsets, fill_value=np.nan):
    _mom = lambda s: s if s > 0 else 0
    _non = lambda s: s if s < 0 else None

    res = np.full_like(a, fill_value=fill_value)
    res[tuple(np.s_[_mom(o):_non(o)] for o in offsets)] \
        = a[tuple(np.s_[_mom(-o):_non(-o)] for o in offsets)]

    return res

# see https://stackoverflow.com/a/27087513/11934495
#@numba.jit
def shift(a: np.typing.NDArray, offsets) -> np.typing.NDArray:
    _mom = lambda s: s if s > 0 else 0
    _non = lambda s: s if s < 0 else None
    return a[tuple(np.s_[_mom(o):_non(o)] for o in offsets)]

#@numba.jit
def slice_like(a: np.typing.NDArray) -> np.s_:
    s = a.shape
    return np.s_[tuple(np.s_[:l] for l in s)]

class RGBAChannel(enum.IntEnum):
    R = 0
    G = 1
    B = 2
    A = 3

@numba.jit
def rgba_frac_to_rgba(rgba: np.typing.NDArray):
    return np.multiply(rgba, 255).astype(np.uint8)

@numba.jit
def rgb_apply_a(
    rgb: np.typing.NDArray,
    a: np.typing.NDArray
):
    return (a / 255. * rgb).astype(np.uint8)

#@numba.jit
def rgba_to_rgb(
    rgba: np.typing.NDArray, 
    rgb_background: np.typing.NDArray
) -> np.typing.NDArray:
    chan = RGBAChannel

    return (
        rgb_apply_a(
            rgb=rgba[..., np.array([chan.R, chan.G, chan.B])], 
            a=rgba[..., np.array([chan.A])]
        ) + rgb_apply_a(
            rgb=rgb_background, 
            a=255 - rgba[..., np.array([chan.A])]
        )             
    )

#@numba.jit
def rgb_to_bilevel(
    rgb: np.typing.NDArray, 
    rgb_background: np.typing.NDArray
) -> np.typing.NDArray:
    rgb = rgb.astype(np.uint8)
    rgb_background = rgb_background.astype(np.uint8)
    
    return (rgb != rgb_background).all(axis=-1)

class Coordinate(typing.NamedTuple):
    x: int
    y: int

    @classmethod
    def make(cls, x, y):
        return cls(x=int(x), y=int(y))

    def transpose(self) -> Coordinate:
        return Coordinate(x=self.y, y=self.x)

@functools.total_ordering
class Dimension(typing.NamedTuple):
    width: int
    height: int

    @classmethod
    def make(cls, width, height) -> Dimension:
        return cls(width=int(width), height=int(height))

    def transpose(self) -> Dimension:
        return Dimension(width=self.height, height=self.width)

    def area(self):
        return self.width * self.height

    def __eq__(self, other: Dimension) -> bool:
        return (self.width == other.width 
            and self.height == other.height)

    def __contains__(self, other: Dimension) -> bool:
        return (self.width >= other.width 
            and self.height >= other.height)

class BBox(typing.NamedTuple):
    left: typing.Any
    top: typing.Any
    right: typing.Any
    bottom: typing.Any

    @classmethod
    def from_dimension(
        cls, 
        dim: Dimension, 
        offset: Coordinate=Coordinate(0, 0)
    ) -> BBox:
        return cls(
            left=offset.x,
            top=offset.y,
            right=offset.x + dim.width,
            bottom=offset.y + dim.height
        )
    
    def dimension(self) -> Dimension:
        return Dimension.make(
            width=self.right - self.left, 
            height=self.bottom - self.top
        )

    def __contains__(self, other: BBox) -> bool:
        return (
            (self.left <= other.left
                and self.top <= other.top)
            and (self.right >= other.right
                and self.bottom >= other.bottom)
        )

BilevelData = np.typing.NDArray[np.bool_]

# see https://en.wikipedia.org/wiki/Summed-area_table
class SummedAreaTable:
    @staticmethod
    #@numba.jit
    def _make(a: np.typing.NDArray):
        return np.apply_over_axes(
            np.cumsum, 
            a, 
            # NOTE outer to inner axis for speed?
            axes=np.arange(2)[::-1]
        )

    def __init__(self, a: np.typing.NDArray):
        self.base = self._make(a)

    @staticmethod
    @numba.jit
    def _area(
        a: np.typing.NDArray, 
        offset: Coordinate, 
        block_size: Dimension
    ):
        x, y = offset
        x_block, y_block = block_size

        return (
            (a[x + x_block, y + y_block] - a[x + x_block, y])
                - (a[x, y + y_block] - a[x, y])
        )

    def area(
        self,
        offset: Coordinate,
        block_size: Dimension
    ):
        return self._area(self.base, offset, block_size)

    def walk(
        self,
        block_size: Dimension
    ):
        x_max, y_max = self.base.shape
        x_block, y_block = block_size

        for x in range(x_max - x_block):
            for y in range(y_max - y_block):
                offset = Coordinate(x, y)
                yield offset, self.area(
                    offset=offset, 
                    block_size=block_size
                )

    @staticmethod
    @numba.jit
    def _area_matrix(
        a: np.typing.NDArray, 
        block_size: Dimension
    ):
        x_block, y_block = block_size

        return (
            (a[x_block:, y_block:] - a[x_block:, :-y_block])
                - (a[:-x_block, y_block:] - a[:-x_block, :-y_block])
        )    

    def area_matrix(
        self,
        block_size: Dimension
    ):
        return self._area_matrix(
            self.base, 
            block_size=block_size
        )

    @staticmethod
    #@numba.jit
    def _paste(
        a: np.typing.NDArray, 
        position: Coordinate, 
        a_src: np.typing.NDArray
    ):
        a_ = shift(a, position)
        s_pad = np.subtract(a_.shape, a_src.shape)
        a_[...] += np.pad(
            a_src, 
            pad_width=[(0, pad_r) for pad_r in s_pad], 
            mode='edge'
        )

        return a

    def paste(
        self,
        position: Coordinate,
        source: SummedAreaTable
    ):
        self.base = self._paste(
            self.base, 
            position=position, 
            a_src=source.base
        )

        return self

        # TODO rm!!!
        # TODO rm debug
        # TODO handle overflow!!!
        print(
            'paste',
            'base:',
            self.base.shape,
            position,
            shift(self.base, position).shape, 
            'source:',
            source.base.shape,
            shift(self.base, position)[slice_like(source.base)].shape
        )
        shift(self.base, position)[slice_like(source.base)] += source.base
        return self

    # TODO test!!!!!!!!!!!!!!!!!!!!!!!!!
    def _rm_area_matrix_comp(self, block_size: Dimension):
        x_max, y_max = self.base.shape
        x_block, y_block = block_size

        res = np.full(((x_max - x_block), (y_max - y_block)), fill_value=np.nan)
        for x in range(x_max - x_block):
            for y in range(y_max - y_block):
                offset = Coordinate(x, y)
                res[x, y] = self.area(
                    offset=offset, 
                    block_size=block_size
                )

        return res
