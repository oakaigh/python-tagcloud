import numpy
import numpy.typing

class ArrayExpression:
    def __getitem__(self, *args, **kwargs):
        return numpy.asarray(*args, **kwargs)
a_ = ArrayExpression()
