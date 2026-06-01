from __future__ import annotations

import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*isinstance\(treespec, LeafSpec\).*is deprecated.*",
)

# Torch imports this optional FLOP-counter module in each Windows dataloader
# worker; Triton is not used by this project, so the warning is just noise.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
