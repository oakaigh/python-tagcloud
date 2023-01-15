import numpy as np
import numpy.typing

import numba


# see https://stackoverflow.com/a/27087513/11934495
#@numba.jit(nopython=True, nogil=True, fastmath=True)
def shift(
    a: np.typing.NDArray, 
    offsets: np.typing.ArrayLike
) -> np.typing.NDArray:
    return a[
        tuple(np.s_[
            (o if o > 0 else 0)
                :(o if o < 0 else None)
        ] for o in offsets)
    ]

# see https://stackoverflow.com/a/27087513/11934495
def _TODO_shift_full(a, offsets, fill_value=np.nan):
    _mom = lambda s: s if s > 0 else 0
    _non = lambda s: s if s < 0 else None

    res = np.full_like(a, fill_value=fill_value)
    res[tuple(np.s_[_mom(o):_non(o)] for o in offsets)] \
        = a[tuple(np.s_[_mom(-o):_non(-o)] for o in offsets)]

    return res


def slice_like(a: np.typing.NDArray) -> np.s_:
    s = a.shape
    return np.s_[tuple(np.s_[:l] for l in s)]
