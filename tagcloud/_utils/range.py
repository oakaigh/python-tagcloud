import typing


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
