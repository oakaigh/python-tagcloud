import typing


def subdict(
    o: dict, 
    keys, 
    ignore_nonexist=False, 
    access_fn=lambda o, k: o[k]
) -> dict:
    return {
        k: access_fn(o, k)
            for k in keys 
            if (k in o if ignore_nonexist else True)
    }

def setattrs(o, **attrs):
    for k, v in attrs.items():
        setattr(o, k, v)

def delattrs(o, *attrs):
    for k in attrs:
        delattr(o, k)

def getattrs(o, *attrs):
    return {k: getattr(o, k) for k in attrs}
        
import contextlib

# see https://stackoverflow.com/a/38532086/11934495
@contextlib.contextmanager
def context_setattrs(o, **attrs):
    noattrs_orig = set(filter(
        lambda k: not hasattr(o, k), 
        attrs.keys()
    ))
    attrs_orig = getattrs(
        o, *(set(attrs.keys()) - noattrs_orig)
    )

    setattrs(o, **attrs)
    yield

    setattrs(o, **attrs_orig)
    delattrs(o, *noattrs_orig)


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


class Range:
    def __init__(
        self, 
        start: typing.Any, stop: typing.Any, 
        step: typing.Any=None,
        include: typing.List[typing.Literal['start', 'stop']]=['start']
    ):
        self.start = start
        self.stop = stop
        self.step = step
        self.include = include

    @staticmethod
    def _lteq(a, b, eq) -> bool: 
        return (a <= b) if eq else (a < b)

    def _is_after_start(self, n) -> bool:
        return self._lteq(self.start, n, 'start' in self.include)

    def _is_before_stop(self, n) -> bool:
        return self._lteq(n, self.stop, 'stop' in self.include)

    def __contains__(self, n) -> bool:
        return (
            (
                self._is_after_start(n)
                    and self._is_before_stop(n)
            ) and (
                n % self.step == 0 
                    if self.step is not None else 
                True
            )
        )

    def __iter__(self):
        if self.step is None:
            raise TypeError('''Range without 'step' is not iterable''')

        raise NotImplementedError()

        # TODO
        '''
        value = self.start
        if self.step > 0:
            while value < self.stop:
                yield value
                value += self.step
        else:
            while value > self.stop:
                yield value
                value += self.step
        '''

import time
import datetime

# see https://stackoverflow.com/a/69156219/11934495
class perf_timer:
    def __init__(self, callback):
        self._callback = callback

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *_args, **_kwargs):
        return self._callback.__call__(
            datetime.timedelta(
                seconds=time.perf_counter() - self._start_time
            )
        )
