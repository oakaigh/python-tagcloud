from . import backend_base
from . import graphics

import typing

import numpy as np

import matplotlib as mpl
import matplotlib.backends.backend_agg


class CanvasMPL(backend_base.CanvasBase):
    def __init__(
        self, 
        backend: typing.Type[mpl.figure.FigureCanvasBase]
            =mpl.backends.backend_agg.FigureCanvas,
        figure: mpl.figure.FigureBase=None
    ):
        super().__init__()
        self._base = backend(figure=figure)

    @property
    def _figure(self):
        return self._base.figure

    @property
    def _background_rgb(self) -> np.typing.NDArray:
        r, g, b, _ = graphics.rgba_frac_to_rgba(
            np.asarray(self._figure.get_facecolor()), 
        )
        return np.asarray((r, g, b))

    # NOTE dimension follows the ones in `data_bool`
    @property
    def dimension(self) -> graphics.Dimension:
        _, _, width, height = self._figure.get_window_extent(
            renderer=self._base.get_renderer()
        ).bounds
        return graphics.Dimension.make(width=height, height=width)

    @property
    def data_rgba(self) -> np.typing.NDArray:
        return np.asarray(
            self._base.copy_from_bbox(self._figure.bbox),
            dtype=np.uint8
        )

    @property
    def data_rgb(self) -> np.typing.NDArray:
        return graphics.rgba_to_rgb(
            self.data_rgba, 
            rgb_background=self._background_rgb
        )

    @property
    def data_bool(self) -> graphics.BilevelData:
        return graphics.rgb_to_bilevel(
            self.data_rgb, 
            rgb_background=self._background_rgb
        )

    # position: coordinate in `data_bool`
    # dimension: follows coordinate convention in `data_bool`

    # NOTE coordinate convention: xy reversed!
    def text(self, text_spec: backend_base.TextSpec) -> graphics.Dimension:
        def _draw(f: mpl.figure.Figure, render: bool):
            if not render:
                return f.draw_without_rendering()
            return f.draw(renderer=f.canvas.get_renderer())

        def _get_extent(t: mpl.text.Text):
            b = t.get_bbox_patch()
            if b is not None:
                return b.get_window_extent()
            return t.get_window_extent(
                renderer=t.get_figure().canvas.get_renderer()
            )

        pos = text_spec.position
        t = self._figure.add_artist(
            mpl.text.Text(
                text=text_spec.content,
                size=text_spec.size,
                rotation=text_spec.rotation,
                transform=None  # do not translate xy coords
            )
        )

        # TODO offset
        _, _, width_f, height_f = self._figure.bbox.bounds        
        _, _, width, height = _get_extent(t).bounds

        if text_spec.position is not None:
            t.set(
                # TODO NOTE invert y axis!!!!
                x=pos.y, y=height_f - height - pos.x,
                #x=pos.y, y=pos.x
            )
            
            _draw(self._figure, render=True)

            # TODO rm
            print('text', pos.y, height_f - height - pos.x)
            print('text', text_spec, _get_extent(t).bounds)   

            # TODO updated region
            self.callbacks.region_update.__call__(self, text_spec.position, None)
        else:
            # hidden
            t.remove()

        return graphics.Dimension.make(width=height, height=width)
