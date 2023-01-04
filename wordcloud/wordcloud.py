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
        mask: typing.Union[np.typing.ArrayLike, None]=None
    ):
        self._canvas = canvas
        self._mask = mask

        self._data = None
        def _f_update(*args, **kwargs):
            d = self._canvas.data_bilevel
            if self._mask is not None:
                d += self._mask
            self._data = graphics.SummedAreaTable(d.astype(np.uint))
        _f_update()
        self._canvas.callbacks.region_update[self] = _f_update

    def __del__(self):
        self._canvas.callbacks.region_update.pop(self)

    @property
    def canvas(self) -> backend_base.CanvasBase:
        return self._canvas

    @property
    def data(self) -> graphics.SummedAreaTable:
        return self._data

    def query_position(
        self, 
        block_size: graphics.Dimension
    ) -> typing.Iterator[graphics.Coordinate]:
        for pos, area in self.data.walk(block_size):
            # unoccupied area
            if not area:
                yield pos
    
    def positions(
        self,
        block_size: graphics.Dimension
    ) -> np.typing.NDArray:
        a = self.data.area_matrix(block_size)
        # unoccupied area == 0
        return np.argwhere(a == 0)

    def sample_position(
        self, 
        block_size: graphics.Dimension, 
        random_state: random.Random
    ) -> graphics.Coordinate:
        r = self.positions(block_size)
        if r.size == 0:
            return None
        return graphics.Coordinate(*random_state.choice(r))


class TextPlacement:
    def __init__(self, 
        canvas: backend_base.CanvasBase, 
        occupancy: OccupancyMap, 
        random_state: random.Random
    ):
        self.canvas = canvas
        self.occupancy = occupancy
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
        size_min, size_max = size_range
        rotation_min, rotation_max = rotation_range

        def _impl(size: float, rotation: float):
            if size is None:
                return None

            if not (size_min <= size and size <= size_max):
                return None

            if not (rotation_min <= rotation and rotation <= rotation_max):
                return None
            
            dim = canvas.text(backend_base.TextSpec(
                content=text, 
                size=size, 
                position=None, 
                rotation=rotation
            ))

            # try to find a position
            pos = occupancy.sample_position(dim, random_state=random_state)
            if pos is not None:
                return backend_base.TextSpec(
                    content=text,
                    size=size, 
                    position=pos, 
                    rotation=rotation
                )

            # if we didn't find a place...
            # first try to rotate!
            res = _impl(
                size=size, 
                rotation=rotation + rotation_step
            )
            if res is not None:
                return res

            # make font smaller
            res = _impl(
                size=size - size_step, 
                rotation=rotation
            )
            if res is not None:
                return res

            return None

        rotation = rotation_min
        if random_state.random() < rotation_prob:
            rotation = random_state.choice(
                range(
                    rotation_min, 
                    rotation_max + rotation_step, 
                    rotation_step
                )
            )
        
        return _impl(size=size_max, rotation=rotation)






import collections.abc

class FrequencyData(typing.NamedTuple):
    token: typing.Any
    frequency: float

class DescendingFrequencyTable:
    @staticmethod
    def _sorted(a: collections.abc.Iterable[FrequencyData]):
        return sorted(
            a, 
            key=lambda x: x.frequency, 
            reverse=True
        )

    @classmethod
    def from_iter(cls, a: collections.abc.Iterable):
        return cls(map(lambda x: FrequencyData(**x), a))

    try:
        import pandas as pd

        @classmethod
        def from_dataframe(cls, df: pd.DataFrame):
            return cls.from_iter(df.to_dict('records'))

        @classmethod
        def from_data(cls, a: collections.abc.Iterable):
            df_token_freqs = cls.pd.Series(a).value_counts()

            return cls.from_dataframe(
                cls.pd.DataFrame({
                    'token': df_token_freqs.index,
                    'frequency': df_token_freqs.values
                })
            )
    except ModuleNotFoundError:
        pass 

    def __init__(self, a: collections.abc.Iterable[FrequencyData]):
        self.base = self._sorted(a)

    def head(self, n):
        inst = self.__new__(self.__class__)
        inst.base = self.base[:n]
        return inst

    @property
    def items(self):
        return self.base



class WordCloud:
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
        random_state = self.random_state

        occupancy = OccupancyMap(canvas, mask=bool_mask)
        text_placement = TextPlacement(
            canvas=canvas, 
            occupancy=occupancy, 
            random_state=random_state
        )

        last_freq = None

        for token, freq in frequency_table.items:
            if freq == 0:
                continue

            # select the text size
            text_size_min, text_size_max = text_props['size_min'], text_props['size_max']
            text_size_scaling = text_props['size_rescaling']
            if last_freq is not None and text_size_scaling != 0:
                text_size_max *= (
                    text_size_scaling * (freq / float(last_freq))
                        + (1 - text_size_scaling)
                )

            text_spec = text_placement.add(
                text=token, 
                size_range=(text_size_min, text_size_max), 
                size_step=text_props['size_step'], 
                rotation_range=text_props['rotation_range'],
                rotation_step=text_props['rotation_step'],
                rotation_prob=text_props['rotation_prob']
            )
            # we were unable to draw any more
            if text_spec is None:
                break

            # draw the text
            canvas.text(text_spec)
            yield text_spec

            last_freq = freq

            # TODO rm debug
            continue
            fig, ax = plt.subplots(2)
            ax[0].imshow(text_placement.occupancy.data.base.astype(np.bool_))
            ax[1].imshow(canvas.data_bilevel)
            plt.suptitle(token)
            plt.show()

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
                rotation_prob=.5
            ),
            **text_props
        }

        def _make_canvas():
            return self.canvas_backend(**canvas_props)

        def _find_text_size_max(n_samples):
            nonlocal self, frequency_table, canvas_props

            canvas = _make_canvas()

            # maximum text size when horizontal
            size_max = canvas.dimension.height

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
        