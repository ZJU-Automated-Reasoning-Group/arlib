from typing import Collection, Callable


def identity(_object):
    return _object


def for_each(_collection: Collection, fn: Callable):
    for item in _collection:
        fn(item)


def get_by_indexes(_collection: Collection, indexes: Collection):
    return [item for i, item in enumerate(_collection) if i in indexes]


def trim(_collection: Collection, condition: Callable = identity):
    return [item for item in _collection if condition(item)]


def trim_by_indexes(_collection: Collection, indexes: Collection):
    return [item for i, item in enumerate(_collection) if i not in indexes]

