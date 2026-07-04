"""
MCP (Model Context Protocol) Gateway for ChatGPT Web -> API
Exposes JSON-RPC endpoint compatible with MCP clients.
"""
import json
from typing import Any, Dict

from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

from utils.Logger import logger

MCP_VERSION = "2024-11-05"
JSONRPC_VERSION = "2.0"


def mcp_error(code: int, message: str, id: Any = None) -> Dict:
    return {"jsonrpc": JSONRPC_VERSION, "id": id, "error": {"code": code, "message": message}}


def mcp_response(result: Any, id: Any = None) -> Dict:
    return {"jsonrpc": JSONRPC_VERSION, "id": id, "result": result}


MCP_TOOLS = [
    {
        "name": "chat_completion",
        "description": "Send a chat completion request to ChatGPT via the web API proxy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {"type": "array", "description": "OpenAI-format messages", "items": {"type": "object"}},
                "model": {"type": "string", "description": "Model name", "default": "gpt-4o"},
                "temperature": {"type": "number", "default": 0.7},
                "max_tokens": {"type": "integer", "default": 4096},
                "tools": {"type": "array", "items": {"type": "object"}}
            },
            "required": ["messages"]
        }
    },
    {
        "name": "generate_image",
        "description": "Generate an image using DALL-E via ChatGPT",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Image prompt"},
                "size": {"type": "string", "enum": ["1024x1024", "1792x1024", "1024x1792"], "default": "1024x1024"}
            },
            "required": ["prompt"]
        }
    }
]


def register_mcp_endpoints(app, api_prefix: str, security_scheme):
    prefix = f"/{api_prefix}/mcp" if api_prefix else "/mcp"

    @app.post(f"{prefix}")
    async def mcp_handler(request: Request, credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(mcp_error(-32700, "Parse error"), status_code=400)

        method = body.get("method")
        params = body.get("params", {})
        req_id = body.get("id")

        if body.get("jsonrpc") != JSONRPC_VERSION:
            return JSONResponse(mcp_error(-32600, "Invalid Request", req_id))

        try:
            if method == "initialize":
                return JSONResponse(mcp_response({
                    "protocolVersion": MCP_VERSION,
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                    "serverInfo": {"name": "ChatGPT2API-MCP", "version": "1.0.0"}
                }, req_id))

            elif method == "tools/list":
                return JSONResponse(mcp_response({"tools": MCP_TOOLS}, req_id))

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})

                if tool_name == "chat_completion":
                    from api.function_calling import inject_tools_into_messages
                    from chatgpt.ChatService import ChatService

                    messages = tool_args.get("messages", [])
                    model = tool_args.get("model", "gpt-4o")
                    tools = tool_args.get("tools")

                    if tools:
                        messages = inject_tools_into_messages(messages, tools)

                    request_data = {
                        "model": model,
                        "messages": messages,
                        "temperature": tool_args.get("temperature", 0.7),
                        "max_tokens": tool_args.get("max_tokens", 4096),
                        "stream": False
                    }

                    req_token = credentials.credentials
                    chat_service = ChatService(req_token)
                    try:
                        await chat_service.set_dynamic_data(request_data)
                        await chat_service.get_chat_requirements()
                        await chat_service.prepare_send_conversation()
                        res = await chat_service.send_conversation()

                        content = ""
                        if isinstance(res, dict):
                            content = res["choices"][0]["message"]["content"]
                        else:
                            async for chunk in res:
                                if chunk.startswith("data: ") and "data: [DONE]" not in chunk:
                                    try:
                                        data = json.loads(chunk[6:])
                                        content += data["choices"][0].get("delta", {}).get("content", "")
                                    except Exception:
                                        pass

                        return JSONResponse(mcp_response({"content": [{"type": "text", "text": content}]}, req_id))
                    finally:
                        await chat_service.close_client()

                else:
                    return JSONResponse(mcp_error(-32601, f"Tool not found: {tool_name}", req_id))

            elif method == "resources/list":
                return JSONResponse(mcp_response({"resources": []}, req_id))

            elif method == "prompts/list":
                return JSONResponse(mcp_response({"prompts": []}, req_id))

            elif method == "notifications/initialized":
                return JSONResponse(mcp_response({}, req_id))

            else:
                return JSONResponse(mcp_error(-32601, f"Method not found: {method}", req_id))

        except Exception as e:
            logger.error(f"MCP error: {str(e)}")
            return JSONResponse(mcp_error(-32603, f"Internal error: {str(e)}", req_id))

    logger.info(f"MCP endpoint registered at {prefix}")
