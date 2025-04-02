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

import os
import dotenv
import anthropic


from compute_eval.models.model_interface import ModelInterface, get_parameter_value


class ClaudeModel(ModelInterface):
    """
    Generate code completions using Clade models.

    Args:
        base_url (str): Base URL for the OpenAI API model.
        model_name (str): Name of the model to use for generating completions.
    """

    def __init__(self, model_name):
        dotenv.load_dotenv()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key is None:
            raise Exception("ANTHROPIC_API_KEY is missing from the .env file.")

        self.model_name = model_name

    def generate_response(self, system_prompt, prompt, params):
        """
        Generate code completions by communicating with the Claude API.

        Args:
            system_prompt (str, optional): The system prompt to use for generating completions.
            problem (dict): The dictionary containing the problem prompt.
            model_type (str): The type of the model ("instruct" or "base").
            temperature (float): Temperature for sampling.
            max_tokens (int): Maximum tokens to generate.

        Returns:
            str: Generated code completion.
        """

        messages = []

        messages.append({"role": "user", "content": prompt})

        client = anthropic.Anthropic()

        try:
            response = client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                temperature=get_parameter_value("temperature", params, 0.2),
                top_p=get_parameter_value("top_p", params, 0.95),
                max_tokens=get_parameter_value("max_tokens", params, 2048),
                stream=False,
            )
        except response.exceptions.RequestException as e:
            if response.status_code == 400:
                raise Exception(
                    "Invalid request was made. Check the headers and payload"
                )
            elif response.status_code == 401:
                raise Exception(
                    "Unauthorized HTTP request. Check your headers and API key"
                )
            elif response.status_code == 403:
                raise Exception("You are forbidden from accessing this resource")
            elif response.status_code > 400:
                raise Exception(
                    "An error occurred when accessing the model API. Check your headers and payload"
                )

        try:
            completion = response.content
        except AttributeError as e:
            print(
                f"WARNING: The completion object is invalid. Could not access 'content' attribute: {str(e)}"
            )
            completion = ""
        except Exception as e:
            raise Exception(f"There was an error when accessing the completion")

        return completion[0].text
