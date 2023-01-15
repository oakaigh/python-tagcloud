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
