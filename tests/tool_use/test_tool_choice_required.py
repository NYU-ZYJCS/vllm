这个文件改动比较多，直接全选替换最快。`Ctrl+A` 全选，删掉，粘贴以下内容：

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from copy import deepcopy

import pytest
import regex as re
from pydantic import TypeAdapter

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.streaming import extract_required_tool_call_streaming
from vllm.tool_parsers.utils import get_json_schema_from_tools

pytestmark = pytest.mark.cpu_test

EXAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for"
                        ", e.g. 'San Francisco'",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get the weather forecast for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the forecast for, e.g. "
                        "'New York'",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to get the forecast for (1-7)",
                    },
                },
                "required": ["city", "days"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
]


def _compile_and_check(
    tools: list[ChatCompletionToolsParam], sample_output, should_match: bool
):
    schema = get_json_schema_from_tools(tools=tools, tool_choice="required")
    assert isinstance(schema, dict)

    from outlines_core.json_schema import build_regex_from_schema

    regex = build_regex_from_schema(json.dumps(schema))
    compiled = re.compile(regex)
    matches = compiled.fullmatch(json.dumps(sample_output)) is not None

    assert matches == should_match


VALID_TOOL_OUTPUTS = [
    ([{"name": "get_current_weather", "parameters": {"city": "Vienna"}}], True),
    (
        [
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
            {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
        ],
        True,
    ),
    ([{"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}}], True),
    (
        [
            {"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
        ],
        True,
    ),
    (
        [
            {"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
            {"name": "get_forecast", "parameters": {"city": "Berlin", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
        ],
        True,
    ),
]

VALID_TOOLS = [t[0] for t in VALID_TOOL_OUTPUTS]


@pytest.mark.parametrize(
    "sample_output, should_match",
    VALID_TOOL_OUTPUTS
    + [
        (None, False),
        ([], False),
        ({}, False),
        ([{}], False),
        (
            [
                {
                    "name": "get_current_weather"
                }
            ],
            False,
        ),
        (
            [
                {
                    "name": "get_current_weather",
                    "parameters": {},
                }
            ],
            False,
        ),
        (
            [
                {
                    "name": "get_current_weather",
                    "parameters": None,
                }
            ],
            False,
        ),
        (
            {
                "name": "get_current_weather",
                "parameters": {"city": "Vienna"},
            },
            False,
        ),
        (
            [
                {
                    "name": "get_current_weather",
                    "parameters": {"city": "Vienna", "extra": "value"},
                }
            ],
            False,
        ),
        (
            [
                {
                    "parameters": {"city": "Vienna"},
                    "name": "get_current_weather",
                }
            ],
            False,
        ),
        (
            [
                {
                    "name": "get_forecast",
                    "parameters": {"city": "Vienna"},
                }
            ],
            False,
        ),
        (
            [{"name": "get_weather", "parameters": {"city": "Vienna", "days": 7}}],
            False,
        ),
        (
            [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}, {}],
            False,
        ),
    ],
)
def test_structured_outputs_json(sample_output, should_match):
    _compile_and_check(
        tools=TypeAdapter(list[ChatCompletionToolsParam]).validate_python(
            EXAMPLE_TOOLS
        ),
        sample_output=sample_output,
        should_match=should_match,
    )


def update_parameters_none(tool: ChatCompletionToolsParam) -> ChatCompletionToolsParam:
    tool.function.parameters = None
    return tool


def update_parameters_empty_dict(
    tool: ChatCompletionToolsParam,
) -> ChatCompletionToolsParam:
    tool.function.parameters = {}
    return tool


@pytest.mark.parametrize(
    "sample_output, should_match",
    [
        (None, False),
        ([], False),
        ({}, False),
        ([{}], False),
        (
            [
                {
                    "name": "get_current_weather"
                }
            ],
            False,
        ),
        (
            [
                {
                    "name": "get_current_weather",
                    "parameters": None,
                }
            ],
            False,
        ),
        (
            [
                {
                    "name": "get_current_weather",
                    "parameters": {"extra": "value"},
                }
            ],
            False,
        ),
        (
            [
                {
                    "name": "get_current_weather",
                    "parameters": {},
                }
            ],
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "update_parameters", [update_parameters_none, update_parameters_empty_dict]
)
def test_structured_outputs_json_without_parameters(
    sample_output, should_match, update_parameters
):
    updated_tools = [deepcopy(EXAMPLE_TOOLS[0])]
    tools = TypeAdapter(list[ChatCompletionToolsParam]).validate_python(updated_tools)
    tools = list(map(update_parameters, tools))
    assert all(
        [
            tool.function.parameters is None or tool.function.parameters == {}
            for tool in tools
        ]
    )
    _compile_and_check(
        tools=tools, sample_output=sample_output, should_match=should_match
    )


@pytest.mark.parametrize("output", VALID_TOOLS)
@pytest.mark.parametrize("empty_params", [False, True])
@pytest.mark.parametrize("delta_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_streaming_output_valid(output, empty_params, delta_len):
    output = deepcopy(output)
    if empty_params:
        output = [{"name": o["name"], "parameters": {}} for o in output]
    output_json = json.dumps(output)

    previous_text = ""
    function_name_returned = False
    current_tool_array_index = -1
    messages = []
    for i in range(0, len(output_json), delta_len):
        delta_text = output_json[i : i + delta_len]
        current_text = previous_text + delta_text

        delta_message, function_name_returned, current_tool_array_index = (
            extract_required_tool_call_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                function_name_returned=function_name_returned,
                current_tool_array_index=current_tool_array_index,
                tool_call_idx=None,
                tool_call_id_type="random",
            )
        )

        if delta_message:
            messages.append(delta_message)

        previous_text = current_text

    assert len(messages) > 0

    combined_messages = "["
    for message in messages:
        if message.tool_calls[0].function.name:
            if len(combined_messages) > 1:
                combined_messages += "},"

            combined_messages += (
                '{"name": "'
                + message.tool_calls[0].function.name
                + '", "parameters": '
                + message.tool_calls[0].function.arguments
            )
        else:
            combined_messages += message.tool_calls[0].function.arguments
    combined_messages += "}]"
    assert json.loads(combined_messages) == output
    assert json.dumps(json.loads(combined_messages)) == output_json


def test_streaming_output_valid_with_trailing_extra_data():
    output = [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}]
    output_json = json.dumps(output) + "\nDONE"

    previous_text = ""
    function_name_returned = False
    current_tool_array_index = -1
    messages = []
    delta_len = 3
    for i in range(0, len(output_json), delta_len):
        delta_text = output_json[i : i + delta_len]
        current_text = previous_text + delta_text

        delta_message, function_name_returned, current_tool_array_index = (
            extract_required_tool_call_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                function_name_returned=function_name_returned,
                current_tool_array_index=current_tool_array_index,
                tool_call_idx=None,
                tool_call_id_type="random",
            )
        )

        if delta_message:
            messages.append(delta_message)

        previous_text = current_text

    assert len(messages) > 0


@pytest.mark.parametrize("delta_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_streaming_multiple_tool_calls_each_gets_header(delta_len):
    """Each tool call in a multi-tool required stream must emit its own header.

    Regression test for the bug where a single stream-level boolean
    ``function_name_returned`` was shared across all tool calls, causing
    the second (and subsequent) tool calls to skip their name/id header.
    """
    output = [
        {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
        {"name": "get_forecast", "parameters": {"city": "Berlin", "days": 3}},
    ]
    output_json = json.dumps(output)

    previous_text = ""
    function_name_returned = False
    current_tool_array_index = -1
    messages = []

    for i in range(0, len(output_json), delta_len):
        delta_text = output_json[i : i + delta_len]
        current_text = previous_text + delta_text

        delta_message, function_name_returned, current_tool_array_index = (
            extract_required_tool_call_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                function_name_returned=function_name_returned,
                current_tool_array_index=current_tool_array_index,
                tool_call_idx=None,
                tool_call_id_type="random",
            )
        )

        if delta_message:
            messages.append(delta_message)
        previous_text = current_text

    named_messages = [m for m in messages if m.tool_calls[0].function.name]
    assert len(named_messages) == len(output), (
        f"Expected {len(output)} tool-call headers, got {len(named_messages)}. "
        "Each tool call must emit its own name/id header."
    )
    emitted_names = [m.tool_calls[0].function.name for m in named_messages]
    expected_names = [t["name"] for t in output]
    assert emitted_names == expected_names, (
        f"Tool call names mismatch: {emitted_names} != {expected_names}"
    )

    args_by_index: dict[int, str] = {}
    for m in messages:
        tc = m.tool_calls[0]
        idx = tc.index if tc.index is not None else 0
        args_by_index.setdefault(idx, "")
        if tc.function.arguments:
            args_by_index[idx] += tc.function.arguments

    for i, tool in enumerate(output):
        reconstructed = json.loads(args_by_index[i])
        assert reconstructed == tool["parameters"], (
            f"Tool {i} arguments mismatch: {reconstructed} != {tool['parameters']}"
        )
```

粘贴完 commit，告诉我。
