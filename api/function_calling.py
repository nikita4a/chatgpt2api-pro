"""
Function Calling / Tool Use for ChatGPT Web → API
Uses prompt engineering to emulate native function calling.
Parses model's JSON response and converts to OpenAI-compatible format.
"""
import json
import re
import uuid
import time
from typing import List, Dict, Optional, Any, Tuple

from utils.Logger import logger

TOOL_CALL_SYSTEM_PROMPT = """# Tools

You have access to the following tools:

{tools_json}

## Tool Use Protocol

When you need to use a tool, respond ONLY with a JSON object in the following format, wrapped in ```json code blocks:

```json
{{
  "tool_calls": [
    {{
      "id": "call_<random>",
      "type": "function",
      "function": {{
        "name": "<tool_name>",
        "arguments": "<json_encoded_arguments>"
      }}
    }}
  ]
}}
```

After the tool call, the user will respond with the tool output. Then you can continue the conversation.

IMPORTANT:
- Only call tools when absolutely necessary
- Use proper JSON encoding for arguments
- If no tool is needed, respond normally without JSON
- Do NOT wrap your normal responses in JSON — only tool calls
"""


def build_tool_system_message(tools: List[Dict]) -> Optional[Dict[str, Any]]:
    if not tools:
        return None
    tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
    return {
        "role": "system",
        "content": TOOL_CALL_SYSTEM_PROMPT.format(tools_json=tools_json)
    }


def parse_tool_calls_from_response(content: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    if not content:
        return None, None
    
    json_pattern = r'```json\s*\n(.*?)\n\s*```'
    json_match = re.search(json_pattern, content, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls = parsed["tool_calls"]
                text_before = content[:json_match.start()].strip()
                text_after = content[json_match.end():].strip()
                text_content = (text_before + " " + text_after).strip() or None
                return tool_calls, text_content
        except json.JSONDecodeError:
            pass
    
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            return parsed["tool_calls"], None
    except json.JSONDecodeError:
        pass
    
    return None, content


def format_tool_result_message(tool_call_id: str, tool_name: str, tool_result: str) -> Dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": tool_result
    }


def build_openai_tool_call_response(
    model: str,
    tool_calls: List[Dict],
    text_content: Optional[str] = None,
    finish_reason: str = "tool_calls"
) -> Dict[str, Any]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    
    message: Dict[str, Any] = {"role": "assistant"}
    
    if text_content:
        message["content"] = text_content
    
    if tool_calls:
        openai_tool_calls = []
        for tc in tool_calls:
            args = tc["function"]["arguments"]
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            openai_tc = {
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": args
                }
            }
            openai_tool_calls.append(openai_tc)
        
        message["tool_calls"] = openai_tool_calls
    
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def inject_tools_into_messages(messages: List[Dict], tools: Optional[List[Dict]]) -> List[Dict]:
    if not tools:
        return messages
    
    tool_system_msg = build_tool_system_message(tools)
    if not tool_system_msg:
        return messages
    
    new_messages = []
    system_found = False
    
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = msg["content"] + "\n\n" + tool_system_msg["content"]
            system_found = True
            new_messages.append(msg)
        else:
            new_messages.append(msg)
    
    if not system_found:
        new_messages.insert(0, tool_system_msg)
    
    return new_messages
