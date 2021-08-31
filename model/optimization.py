# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np

from paddle.optimizer.lr import LambdaDecay

__all__ = [
    'ConstDecayWithWarmup'
]


def is_integer(number):
    return isinstance(number, int)


class ConstDecayWithWarmup(LambdaDecay):
    """
    Creates a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate` during warmup periods and keeps learning
    rate a constant after that.

    Args:
        learning_rate (float):
            The base learning rate. It is a python float number.
        warmup (int or float):
            If int, it means the number of steps for warmup. If float, it means
            the proportion of warmup in total training steps.
        total_steps (int, optional):
            The number of training steps. If `warmup` is a float number,
            `total_steps` must be provided.
            Defaults to None.
        last_epoch (int, optional):
            The index of last epoch. It can be set to restart training. If
            None, it means initial learning rate.
            Defaults to -1.

    Examples:

        .. code-block:: python

            from paddlenlp.transformers import ConstScheduleWithWarmup
            lr, warmup_steps = 0.1, 100
            lr_scheduler = ConstScheduleWithWarmup(lr, warmup_steps)

    """

    def __init__(self,
                 learning_rate,
                 warmup,
                 decay_steps=[],
                 total_steps=None,
                 last_epoch=-1,
                 verbose=False):
        if is_integer(warmup):
            warmup_steps = warmup
        elif total_steps:
            warmup_steps = int(math.floor(warmup * total_steps))
        else:
            raise ValueError(
                "Please provide total steps if `warmup` is a float number , or provide integer for argument `warmup`."
            )
        for decay_step in decay_steps:
            if decay_step <= warmup_steps:
                raise ValueError('decay steps should large than the warmup steps')
        self.decay_steps = np.array(decay_steps) if len(decay_steps) > 0 else None

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1.0, warmup_steps))
            elif self.decay_steps is not None:
                return pow(0.1, np.sum(self.decay_steps <= current_step))
            else:
                return 1.0

        super(ConstDecayWithWarmup, self).__init__(learning_rate, lr_lambda, last_epoch, verbose)