import collections as coll
import collections.abc

_builtin_sorted = sorted

def sorted(a: coll.abc.Sequence, ascending=False):
    # TODO NOTE False will sort ascending, True will sort descending. Default is False
    def _impl_range(a: range, ascending=False):
        def _copy(r: range):
            return type(r)(r.start, r.stop, r.step)

        def _reverse(r: range):
            return type(r)(r.stop, r.start, -r.step)

        def _is_ascending(r: range): return r.start < r.stop

        def _is_descending(r: range): return r.start > r.stop

        _should_reverse = (
            _is_ascending if not ascending else
                _is_descending
        )

        if _should_reverse(a):
            return _reverse(a)
        return _copy(a)    

    _impl = _builtin_sorted

    if isinstance(a, range):
        return _impl_range(a)
    return _impl(a, reverse=ascending)

# TODO !!!!!!
def sample(a: coll.abc.Sequence):
    raise NotImplementedError()

# TODO
__all__ = [
    'sorted',
    'sample'
]
