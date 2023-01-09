import typing
import dataclasses as dc
import types
import abc

from . import utils
from . import graphics


@dc.dataclass(frozen=True)
class SpecBase(abc.ABC):
    def set(self, **attrs):
        return dc.replace(self, **attrs)

@dc.dataclass(frozen=True)
class ElementSpec(SpecBase):
    position: graphics.Coordinate
    rotation: int

# NOTE coord system: origin upper left 
class CanvasBase(abc.ABC):
    def __init__(self):
        self.callbacks = types.SimpleNamespace(
            region_update=utils.Callback[
                typing.Callable[
                    [
                        CanvasBase, 
                        graphics.Coordinate, 
                        graphics.BilevelData
                    ], 
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

    @dc.dataclass(frozen=True)
    class TextSpec(ElementSpec):
        content: str
        size: int

    @abc.abstractmethod
    def text(self, text_spec: TextSpec) -> graphics.Dimension:
        raise NotImplementedError()

TextSpec = CanvasBase.TextSpec
