from __future__ import annotations


import typing
import dataclasses as dc
import types
import abc

from . import graphics


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

@dc.dataclass(frozen=True)
class SpecBase(abc.ABC):
    def set(self, **attrs):
        return dc.replace(self, **attrs)

@dc.dataclass(frozen=True)
class BoxSpec(SpecBase):
    rotation: int

@dc.dataclass(frozen=True)
class TextBoxSpec(BoxSpec):
    content: str
    size: int
    # TODO !!!!!!!!!!!!!!!!
    #font: FontBase 

class TextBoxBase(abc.ABC):
    @property
    def dimension(self):
        raise NotImplementedError()

    def render(self, position: graphics.Coordinate):
        raise NotImplementedError()

# NOTE coord system: origin upper left 
class CanvasBase(abc.ABC):
    def __init__(self):
        self.callbacks = types.SimpleNamespace(
            region_update=Callback[
                typing.Callable[
                    [
                        CanvasBase, 
                        graphics.Coordinate, 
                        graphics.BoolData
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
    def data_bool(self) -> graphics.BoolData:
        raise NotImplementedError()

    @abc.abstractmethod
    def textbox(self, textbox_spec: TextBoxSpec) -> TextBoxBase:
        pass



    # TODO deprecate @#######################
    @abc.abstractmethod
    def text(self, text_spec: TextSpec) -> graphics.Dimension:
        raise NotImplementedError()
    ####################################

# TODO deprecate ##########
@dc.dataclass(frozen=True)
class TextSpec:
    position: graphics.Coordinate
    rotation: int    
    content: str
    size: int
# TODO !!!!!!!!!!!!!########

