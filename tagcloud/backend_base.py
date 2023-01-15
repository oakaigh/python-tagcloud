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

class Callback(dict):
    def register(
        self, 
        *callables: typing.Callable, 
        **keyed_callables: typing.Callable
    ):
        self.update(
            map(lambda f: (f, f), callables),
            **keyed_callables
        )
        return self

    # dispatch registered callbacks one by one
    def dispatch(self, *args, **kwargs):
        for f in self.values():
            yield f.__call__(*args, **kwargs)

    # dispatch all of the registered callbacks
    def __call__(self, *args, **kwargs):
        return list(self.dispatch(*args, **kwargs))

# NOTE coord system: origin upper left 
class CanvasBase(abc.ABC):
    def __init__(self):
        self.callbacks = types.SimpleNamespace(
            region_update=Callback[
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
