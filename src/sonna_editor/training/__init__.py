from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*isinstance\\(treespec, LeafSpec\\) is deprecated.*",
    category=DeprecationWarning,
)
