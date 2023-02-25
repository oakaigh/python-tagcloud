from __future__ import annotations


from . import utils

from . import backend_base
from . import graphics


import typing
import dataclasses as dc
import random

import collections as coll
import collections.abc

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
            nonlocal self
            # TODO overlay?
            self._base.paste(
                position,
                graphics.AreaTable(_as_bool(region))
            )

        self._canvas.callbacks.region_update[self] = f

    def __del__(self):
        self._canvas.callbacks.region_update.pop(self)

    @property
    def canvas(self): 
        return self._canvas

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
    @dc.dataclass(frozen=True)
    class Spec:
        position: graphics.Coordinate
        textbox_spec: backend_base.TextBoxSpec
    
    def __init__(self, 
        canvas: backend_base.CanvasBase, 
        mask: np.typing.ArrayLike | None,
        random_state: random.Random
    ):
        self.occupancy = OccupancyMap(canvas, mask=mask)
        self.random_state = random_state

    def add(
        self,
        content: str,
        sizes: coll.abc.Sequence,
        rotations: coll.abc.Sequence,
        rotation_weights: coll.abc.Sequence,
        rotation_prob: float
    ) -> Spec:
        random_state = self.random_state
        occupancy = self.occupancy
        canvas = self.occupancy.canvas

        # TODO!!!!!!!!!!!!!!!!!
        rotation = rotations[0]
        # TODO
        # utils.sequence.sample(rotations, weights=rotation_weights)
        #if random_state.random() < rotation_prob:
        #    # TODO
        #    # NOTE see https://stackoverflow.com/a/11949245/11934495
        #    rotation = random_state.choice(rotations)        

        for s in utils.sequence.sorted(
            sizes, 
            ascending=False
        ):
            textbox_spec = backend_base.TextBoxSpec(
                rotation=rotation,
                content=content, 
                size=s
            )
            textbox = canvas.textbox(textbox_spec)

            pos = occupancy.sample_position(
                textbox.dimension,
                random_state=random_state
            )
            if pos is not None:
                textbox.render(position=pos)
                return self.Spec(
                    position=pos,
                    textbox_spec=textbox_spec
                )

        return None

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
        def from_dataframe(cls, df: pd.DataFrame) -> DescendingFrequencyTable:
            return cls.from_iter(df.to_dict('records'))

        @classmethod
        def from_data(cls, a: typing.Iterable) -> DescendingFrequencyTable:
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
        size_max: int | None
        size_step: int
        size_rescaling: float
        rotation_range: typing.Tuple[int, int]
        rotation_step: float
        rotation_prob: float

    def __init__(
        self, 
        canvas_backend: typing.Type[backend_base.CanvasBase],
        random_state: random.Random=random.Random()
    ):
        self.canvas_backend = canvas_backend
        self.random_state = random_state

    # TODO NOTE mask: boolean mask
    def _generate_layout(
        self,
        frequency_table: DescendingFrequencyTable, 
        canvas: backend_base.CanvasBase,
        bool_mask: typing.Union[np.typing.ArrayLike, None],
        text_props: TextParams
    ) -> typing.Iterator[backend_base.TextSpec]:
        class RelativeScaling:
            def __init__(self, n):
                self._n = n
                self._last_freq = None

            def __call__(self, frequency):
                res = 1.
                if self._last_freq is not None:
                    res = (
                        self._n * (frequency / self._last_freq)
                            + (1 - self._n)
                    )
                self._last_freq = frequency
                return res

        text_placement = TextPlacement(
            canvas=canvas, 
            mask=bool_mask, 
            random_state=self.random_state
        )

        max_freq = frequency_table.max().frequency

        # text sizes
        text_size_min, text_size_max = text_props['size_min'], text_props['size_max']

        # relative scaling
        rscale = RelativeScaling(text_props['size_rescaling'])

        for token, freq in frequency_table.items:
            # normalize frequency
            freq = freq / max_freq

            if freq == 0:
                continue

            text_size_max *= rscale(freq)

            # TODO !!!!!
            text_spec = text_placement.add(
                content=token,
                sizes=range(
                    text_size_min, 
                    int(text_size_max), 
                    text_props['size_step']
                ),
                rotations=range(
                    *text_props['rotation_range'], 
                    text_props['rotation_step']
                ),
                rotation_weights=None,
                rotation_prob=None
            )

            # TODO rm debug
            print('progress', token, freq)

            # we were unable to draw any more
            if text_spec is None:
                break
            yield text_spec

    def draw(
        self,
        frequency_table: DescendingFrequencyTable, 
        canvas_props=dict(),
        bool_mask=None,
        text_props=dict()
    ) -> backend_base.CanvasBase:
        text_props = {
            **dict(
                size_min=5, size_max=None,
                size_step=1,
                size_rescaling=.5,
                rotation_range=(0, 90),
                rotation_step=90,
                rotation_prob=.1
            ),
            **text_props
        }

        def _make_canvas():
            nonlocal self, canvas_props
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
                lambda x: x.textbox_spec.size, 
                self._generate_layout(
                    frequency_table=frequency_table.head(n=n_samples),
                    canvas=canvas,
                    bool_mask=bool_mask,
                    text_props={
                        **text_props,
                        'size_max': size_max
                    }
                )
            ), int)
            
            if len(sizes) < 1:
                raise Exception('canvas out of space')

            if len(sizes) == 1:
                return sizes[0]

            return 2 * np.prod(sizes) / np.sum(sizes)

        if text_props.get('size_max', None) is None:
            text_props['size_max'] = int(_find_text_size_max(
                n_samples=2
            ))

        canvas = _make_canvas()
        layout = list(self._generate_layout(
            frequency_table=frequency_table,
            canvas=canvas,
            bool_mask=bool_mask,
            text_props=text_props
        ))

        # TODO
        return canvas, layout
        