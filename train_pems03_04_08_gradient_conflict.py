#!/usr/bin/env python3
"""Jointly train PEMS03/04/08 and measure shared-gradient conflict frequency.

This entry point reuses the 8M shared QK Transformer backbone.  Each dataset
has its own node embedding and prediction head, while all other parameters
are shared.  The extra task-specific branch brings the exact total to
8,025,532 parameters.
"""

from __future__ import annotations

import sys

import train_pems_joint_gradient_conflict as core


DATASETS = ("pems03", "pems04", "pems08")
DEFAULT_PARAMETER_COUNT = 8_025_532


def main() -> None:
    core.DATASETS = DATASETS
    core.__doc__ = __doc__
    target_was_set = any(
        argument == "--target_parameters" or argument.startswith("--target_parameters=")
        for argument in sys.argv[1:]
    )
    if not target_was_set:
        sys.argv.extend(["--target_parameters", str(DEFAULT_PARAMETER_COUNT)])
    core.main()


if __name__ == "__main__":
    main()
