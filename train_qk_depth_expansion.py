#!/usr/bin/env python3
"""Train one 7/8/9-layer QK Transformer expansion variant from scratch.

For each added depth, the ``unmatched`` model keeps the original 768/3072
width and lets parameters grow.  The ``matched`` model narrows the remaining
dimensions so total parameters stay close to the original six-layer model.

The implementation delegates the already smoke-tested training lifecycle to
``train_qk_depth_ablation.py`` without modifying that earlier experiment.
"""

from __future__ import annotations

import train_qk_depth_ablation as training
from train_qk_depth_ablation import DepthVariant


RECOMMENDED_TRAIN_BATCH_SIZE = 522
RECOMMENDED_EVAL_BATCH_SIZE = 870


EXPANSION_VARIANTS = {
    variant.name: variant
    for variant in (
        DepthVariant(
            "add_last_1_unmatched", -1, "unmatched", 7, 768, 3072
        ),
        DepthVariant(
            "add_last_1_matched", -1, "matched", 7, 720, 2776
        ),
        DepthVariant(
            "add_last_2_unmatched", -2, "unmatched", 8, 768, 3072
        ),
        DepthVariant(
            "add_last_2_matched", -2, "matched", 8, 672, 2608
        ),
        DepthVariant(
            "add_last_3_unmatched", -3, "unmatched", 9, 768, 3072
        ),
        DepthVariant(
            "add_last_3_matched", -3, "matched", 9, 624, 2544
        ),
    )
}


def main() -> None:
    # The shared trainer represents added layers as a negative removed-layer
    # count, so its invariant remains num_layers == 6 - removed_layers.
    training.VARIANTS = EXPANSION_VARIANTS
    training.RECOMMENDED_TRAIN_BATCH_SIZE = RECOMMENDED_TRAIN_BATCH_SIZE
    training.RECOMMENDED_EVAL_BATCH_SIZE = RECOMMENDED_EVAL_BATCH_SIZE
    training.__doc__ = __doc__
    training.main()


if __name__ == "__main__":
    main()
