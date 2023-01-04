import typing
import types
import abc

from . import utils
from . import graphics


# NOTE coord system: origin upper left 
class CanvasBase(abc.ABC):
    class TextSpec(typing.NamedTuple):
        content: str
        size: int
        rotation: float
        position: graphics.Coordinate = None

    def __init__(self):
        self.callbacks = types.SimpleNamespace(
            region_update=utils.Callback[
                typing.Callable[
                    [CanvasBase, graphics.Coordinate, graphics.BilevelData], 
                    typing.Any
                ]
            ]()
        )

    @property
    @abc.abstractmethod
    def dimension(self) -> graphics.Dimension:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def data_bilevel(self) -> graphics.BilevelData:
        raise NotImplementedError()

    @abc.abstractmethod
    def text(self, text_spec: TextSpec) -> graphics.Dimension:
        raise NotImplementedError()

TextSpec = CanvasBase.TextSpec
