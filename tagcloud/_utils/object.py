import contextlib


def setattrs(o, **attrs):
    for k, v in attrs.items():
        setattr(o, k, v)

def delattrs(o, *attrs):
    for k in attrs:
        delattr(o, k)

def getattrs(o, *attrs):
    return {k: getattr(o, k) for k in attrs}

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
