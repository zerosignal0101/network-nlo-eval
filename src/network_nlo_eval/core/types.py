"""自定义类型别名，用于清晰的类型提示."""

import numpy as np
from numpy.typing import NDArray

# NumPy 数组类型别名
NDArrayFloat = NDArray[np.float64]
NDArrayComplex = NDArray[np.complex128]
NDArrayBool = NDArray[np.bool_]
NDArrayInt = NDArray[np.int_]

# Node/Link ID 类型
NodeID = int  # 内部表示的节点ID (0, 1, 2...)
OriginalNodeID = int  # 原始拓扑文件中的节点ID (例如 101, 203)
LinkKey = tuple[NodeID, NodeID]  # 规范化的链路键 (min(u,v), max(u,v))
