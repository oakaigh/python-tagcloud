from __future__ import annotations

from . import backend_base
from . import graphics


import typing

import random

import numpy as np
import numpy.typing


class OccupancyMap:
    def __init__(
        self, 
        canvas: backend_base.CanvasBase, 
        mask: np.typing.ArrayLike=None
    ):
        _as_bool = lambda a: np.asarray(a, dtype=np.bool_)

        self._canvas = canvas
        self._mask = mask

        self._base = graphics.AreaTable(
            _as_bool(self._canvas.data_bool)
                + _as_bool(self._mask)
        )

        def f(_canvas, position, region):
            # TODO overlay?
            self._base.paste(
                position,
                graphics.AreaTable(_as_bool(region))
            )

            # TODO rm debug
            #print('self._base', self._base.base)

            return

        self._canvas.callbacks.region_update[self] = f

    def __del__(self):
        self._canvas.callbacks.region_update.pop(self)

    def positions(
        self,
        block_size: graphics.Dimension
    ) -> np.typing.NDArray:
        # unoccupied area == 0
        return self._base.find(block_size, target_area=0)

    def sample_position(
        self, 
        block_size: graphics.Dimension, 
        random_state: random.Random
    ) -> graphics.Coordinate:
        r = self.positions(block_size)
        if r.size == 0:
            return None

        return graphics.Coordinate(*random_state.choice(r))

    def query_position(
        self, 
        block_size: graphics.Dimension
    ) -> typing.Iterator[graphics.Coordinate]:
        for pos, area in self._base.walk(block_size):
            # unoccupied area
            if area == 0:
                yield pos

class TextPlacement:
    def __init__(self, 
        canvas: backend_base.CanvasBase, 
        mask: np.typing.ArrayLike | None,
        random_state: random.Random
    ):
        self.canvas = canvas
        self.occupancy = OccupancyMap(canvas, mask=mask)
        self.random_state = random_state

    def add(
        self, 
        text: str, 
        size_range: typing.Tuple[float, float], 
        size_step: float, 

        # TODO use range object
        #rotation_range: range
        rotation_range: typing.Tuple[float, float], 
        rotation_step: float, 

        rotation_prob: float,
    ) -> backend_base.TextSpec:
        random_state = self.random_state
        canvas = self.canvas
        occupancy = self.occupancy

        # TODO
        size_min, size_max = size_range
        rotation_min, rotation_max = rotation_range

        def _impl(size: float, rotation: float, epochs_max: int=None):
            if epochs_max is not None:
                if epochs_max <= 0:
                    return None
                epochs_max -= 1

            if size is None:
                return None

            if not (size_min <= size and size <= size_max):
                return None

            if not (rotation_min <= rotation and rotation <= rotation_max):
                return None

            # TODO rm debug
            #print('try', size, rotation)

            text_spec = backend_base.TextSpec(
                position=None, 
                rotation=rotation,
                content=text, 
                size=size
            )
            dim = canvas.text(text_spec)

            # try to find a position
            pos = occupancy.sample_position(dim, random_state=random_state)
            if pos is not None:
                text_spec = text_spec.set(position=pos)
                # draw the text
                canvas.text(text_spec)
                return text_spec

            # TODO rotate only once!!!!!!!!
            # if we didn't find a place...
            # first try to rotate!
            #res = _impl(
            #    size=size, 
            #    rotation=rotation + rotation_step,
            #    epochs_max=1
            #)
            #if res is not None:
            #    return res

            # make font smaller
            res = _impl(
                size=size - size_step, 
                rotation=rotation,
                epochs_max=epochs_max
            )
            if res is not None:
                return res

            return None

        rotation = rotation_min
        #if random_state.random() < rotation_prob:
        #    # TODO
        #    # see https://stackoverflow.com/a/11949245/11934495
        #    rotation = random_state.choice(
        #        range(
        #            rotation_min, 
        #            rotation_max + rotation_step, 
        #            rotation_step
        #        )
        #    )

        # TODO rm debug
        print('try init', size_max, rotation)

        return _impl(size=size_max, rotation=rotation)


class FrequencyData(typing.NamedTuple):
    token: typing.Any
    frequency: float

class DescendingFrequencyTable:
    @staticmethod
    def _sorted(a: typing.Iterable[FrequencyData]) -> typing.Iterable[FrequencyData]:
        return sorted(
            a, 
            key=lambda x: x.frequency, 
            reverse=True
        )

    @classmethod
    def from_iter(cls, a: typing.Iterable) -> DescendingFrequencyTable:
        return cls(map(lambda x: FrequencyData(**x), a))

    try:
        import pandas as pd

        @classmethod
        def from_dataframe(cls, df: pd.DataFrame):
            return cls.from_iter(df.to_dict('records'))

        @classmethod
        def from_data(cls, a: typing.Iterable):
            df_token_freqs = cls.pd.Series(a).value_counts()

            return cls.from_dataframe(
                cls.pd.DataFrame({
                    'token': df_token_freqs.index,
                    'frequency': df_token_freqs.values
                })
            )
    except ModuleNotFoundError:
        pass 

    def __init__(self, a: typing.Iterable[FrequencyData]):
        self.base = self._sorted(a)

    def head(self, n) -> DescendingFrequencyTable:
        inst = self.__new__(self.__class__)
        inst.base = self.base[:n]
        return inst

    @property
    def items(self) -> typing.Iterable[FrequencyData]:
        return self.base

    def max(self) -> FrequencyData:
        return max(self.items, key=lambda x: x.frequency)



class TagCloud:
    class TextParams(typing.TypedDict):
        size_min: int
        size_max: typing.Union[int, None]
        size_step: int
        size_rescaling: float
        rotation_range: typing.Tuple[float, float]
        rotation_step: float
        rotation_prob: float

    def __init__(
        self, 
        canvas_backend: typing.Type[backend_base.CanvasBase],
        random_state: random.Random=random.Random()
    ):
        self.canvas_backend = canvas_backend
        self.random_state = random_state

    # mask: boolean mask
    def _generate_layout(
        self,
        frequency_table: DescendingFrequencyTable, 
        canvas: backend_base.CanvasBase,
        bool_mask: typing.Union[np.typing.ArrayLike, None],
        text_props: TextParams
    ) -> typing.Iterator[backend_base.TextSpec]:
        text_placement = TextPlacement(
            canvas=canvas, 
            mask=bool_mask, 
            random_state=self.random_state
        )

        max_freq = frequency_table.max().frequency
        last_freq = None

        # text sizes
        text_size_min, text_size_max = text_props['size_min'], text_props['size_max']
        text_size_scaling = text_props['size_rescaling']

        for token, freq in frequency_table.items:
            # normalize
            freq = freq / max_freq

            if freq == 0:
                continue

            if last_freq is not None:
                text_size_max *= (
                    text_size_scaling * (freq / last_freq)
                        + (1 - text_size_scaling)
                )
            last_freq = freq

            # TODO rm
            import cProfile
            with cProfile.Profile() as pr:
                text_spec = text_placement.add(
                    text=token, 
                    size_range=(text_size_min, text_size_max), 
                    size_step=text_props['size_step'], 
                    rotation_range=text_props['rotation_range'],
                    rotation_step=text_props['rotation_step'],
                    rotation_prob=text_props['rotation_prob']
                )

                pr.print_stats(sort='time')

            # we were unable to draw any more
            if text_spec is None:
                break
            yield text_spec


            # TODO rm debug
            print('progress', token, freq)
            continue
            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots(2)
            #ax[0].imshow(text_placement.occupancy._base.base.astype(np.bool_))
            #ax[1].imshow(canvas.data_bool)
            #plt.suptitle(token)
            #plt.show()

    def draw(
        self,
        frequency_table: DescendingFrequencyTable, 
        canvas_props=dict(),
        bool_mask=None,
        text_props=dict()
    ) -> backend_base.CanvasBase:
        text_props = {
            **dict(
                size_min=5., size_max=None,
                size_step=1.,
                size_rescaling=.5,
                rotation_range=(0, 90),
                rotation_step=90,
                rotation_prob=.1
            ),
            **text_props
        }

        def _make_canvas():
            return self.canvas_backend(**canvas_props)

        def _find_text_size_max(n_samples):
            nonlocal self, frequency_table, canvas_props

            canvas = _make_canvas()

            # maximum text size when horizontal
            size_max = canvas.dimension.transpose().height

            # we only have one token. We make it big!
            if len(frequency_table.items) == 1:
                return size_max

            sizes = np.fromiter(map(
                lambda x: x.size, 
                self._generate_layout(
                    frequency_table=frequency_table.head(n=n_samples),
                    canvas=canvas,
                    bool_mask=bool_mask,
                    text_props={
                        **text_props,
                        'size_max': size_max
                    }
                )
            ), float)
            
            if len(sizes) < 1:
                raise Exception('canvas out of space')

            if len(sizes) == 1:
                return sizes[0]

            return 2 * np.prod(sizes) / np.sum(sizes)

        if text_props.get('size_max', None) is None:
            text_props['size_max'] = _find_text_size_max(
                n_samples=2
            )

        canvas = _make_canvas()
        layout = list(self._generate_layout(
            frequency_table=frequency_table,
            canvas=canvas,
            bool_mask=bool_mask,
            text_props=text_props
        ))

        # TODO
        return canvas
        