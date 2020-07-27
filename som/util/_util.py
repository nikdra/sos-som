from functools import reduce
from collections import defaultdict


def group_by(key, seq):
    """
    Group a sequence of (key, value) tuples by their respective keys and return a dict with lists of values.

    Parameters
    ----------
    key: function(tuple)
        Usually a lambda expression. Returns the key of the tuple. Example: lambda pair: pair[0].
    seq: array-like of (key, value) tuples
        The array or list containing the tuples that should be grouped together by key.

    Returns
    -------
    grouped_dict: dict
        Dictionary where each entry is a key with a list of values for that key.
    """
    return reduce(lambda grp, val: grp[key(val)].append(val[1]) or grp, seq, defaultdict(list))