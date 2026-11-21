"""Array and network identifier type aliases."""

import numpy as np
from numpy.typing import NDArray

NDArrayFloat = NDArray[np.float64]
NDArrayComplex = NDArray[np.complex128]
NDArrayBool = NDArray[np.bool_]
NDArrayInt = NDArray[np.int_]

NodeID = int  # Internal node identifier.
OriginalNodeID = int  # Identifier in the original topology.
LinkKey = tuple[NodeID, NodeID]  # Ordered for directed links, sorted otherwise.
