from cc_torch import _C


def get_connected_components(x, get_counts=True):
    return _C.cc_2d(x.contiguous(), get_counts)
