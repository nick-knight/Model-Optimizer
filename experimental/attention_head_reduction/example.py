# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal calibration and emulated-compression example."""

import torch

from experimental.attention_head_reduction import HeadProjector, SecondMomentAccumulator


def main() -> None:
    """Calibrate synthetic keys and apply independently ranked projections."""
    accumulator = SecondMomentAccumulator(num_heads=4, head_dim=8)
    for _ in range(4):
        keys = torch.randn(2, 4, 16, 8)
        accumulator.update(keys, head_axis=1)

    projector = HeadProjector.from_second_moment(accumulator.compute(), ranks=[2, 3, 4, 5])
    projected_keys = projector(keys, head_axis=1)
    print(projected_keys.shape, projector.ranks.tolist())


if __name__ == "__main__":
    main()
