# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import fire
import yaml

from compute_eval.evaluation import evaluate_functional_correctness
from compute_eval.generate_completions import generate_samples


def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def generate_samples_with_config(config_file=None, **kwargs):
    if config_file:
        config = load_config(config_file)
        kwargs.update(config)
    generate_samples(**kwargs)


def evaluate_functional_correctness_with_config(config_file=None, **kwargs):
    if config_file:
        config = load_config(config_file)
        kwargs.update(config)
    evaluate_functional_correctness(**kwargs)


def main():
    fire.Fire(
        {
            "evaluate_functional_correctness": evaluate_functional_correctness_with_config,
            "generate_samples": generate_samples_with_config,
        }
    )


if __name__ == "__main__":
    main()
