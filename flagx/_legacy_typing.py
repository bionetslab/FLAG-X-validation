
import sys
from typing import TYPE_CHECKING

# Base typing imports that exist in all versions
from typing import (
    TypeVar,
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    Callable,
    Iterable,
    Sequence,
    Any,
)

# typing imports that are version dependent
if sys.version_info >= (3, 11):
    from typing import Literal, Final, TypedDict, Protocol, Annotated
elif sys.version_info >= (3, 8):
    from typing_extensions import Literal, Final, TypedDict, Protocol, Annotated
elif sys.version_info >= (3, 6):
    from typing_extensions import Literal, Final, TypedDict, Protocol
    Annotated = Any
else:
    Literal = Final = TypedDict = Protocol = Annotated = Any

# Self or TypeVar depending on version
if sys.version_info >= (3, 11):
    from typing import Self as SelfSomClassifier
    from typing import Self as SelfSoftmaxClassifier

    # Todo
    from typing import Self as SelfFoo
    from typing import Self as SelfBar
else:
    SelfSomClassifier = TypeVar('SelfSomClassifier', bound='SomClassifier')
    SelfSoftmaxClassifier = TypeVar('SelfSoftmaxClassifier', bound='SoftmaxClassifier')


__all__ = [
    'TypeVar', 'Optional', 'Union', 'Tuple', 'List', 'Dict', 'Callable', 'Iterable', 'Sequence', 'Any',
    'Literal', 'Final', 'TypedDict', 'Protocol', 'Annotated', 'Optional',
    'SelfSomClassifier', 'SelfSoftmaxClassifier',
]

# === For type checkers (no circular imports at runtime) ===
if TYPE_CHECKING:
    from flagx.gating import SomClassifier
    from flagx.gating import SoftmaxClassifier

