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

from openai import OpenAI


class ModelInterface:
    """
    Base class for generating code completions.
    """

    def generate_response(self, system_prompt, prompt, params):
        """
        Generate code completions by communicating with the OpenAI API.

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

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
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
            completion = response.choices[0].message.content
        except KeyError as e:
            print(
                f"WARNING: The completion object is invalid. Could not find the key {str(e)}"
            )
            completion = ""
        except Exception as e:
            raise Exception(f"There was an error when accessing the completion")

        return completion


def get_parameter_value(parameter, parameters, default_value):
    if parameters is not None and parameter in parameters:
        return parameters[parameter]
    else:
        return default_value
