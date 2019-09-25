from typing import List, TypeVar

from future.moves import itertools
_S = TypeVar('_S')


def flatten(l: List[List[_S]]) -> List[_S]:
    return list(itertools.chain.from_iterable([(i if isinstance(i, list) else [i]) for i in l]))
