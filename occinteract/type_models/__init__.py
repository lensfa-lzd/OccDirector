"""Type Models 模块。

提供项目中使用的类型别名和类型定义。
将嵌套类型按类别组织，便于代码维护和重用。
"""

from .dataset import (
    CaptionInstance,
    OccCaptionDatasetInstance,
    OccInstance,
    OccDatasetInstance,
)

__all__ = [
    # dataset
    "CaptionInstance",
    "OccCaptionDatasetInstance",
    "OccInstance",
    "OccDatasetInstance",
]
