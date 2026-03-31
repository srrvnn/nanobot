"""Gemini LLM Provider using the new google-genai SDK."""

import json
import logging
import uuid
from typing import Any, AsyncGenerator

from google import genai
from google.genai import types

from providers.base import LLMProvider, LLMResponse, ToolCallRequest

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """Google Gemini specific implementation using the native python SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        retry_config: dict[str, Any] | None = None,
        default_model: str = "gemini-3-flash-preview",
    ):
        """Initialize the Gemini Provider."""
        super().__init__(api_key, api_base, retry_config)
        self.default_model = default_model
        
        client_options = {}
        if self.api_base:
            client_options["http_options"] = {"base_url": self.api_base}
            
        self.client = genai.Client(api_key=self.api_key, **client_options)

    def get_default_model(self) -> str:
        """Get the default model string fallback."""
        return self.default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Execute a chat completion request to Gemini."""
        resolved_model = model or self.default_model
        
        # Strip "gemini/" prefix if present
        if resolved_model.startswith("gemini/"):
            resolved_model = resolved_model.split("/", 1)[1]
            
        system_instruction, contents = self._translate_messages(messages, tools)
        genai_tools = self._translate_tools(tools)
        
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
            
        if genai_tools:
            config_kwargs["tools"] = genai_tools
            
        # Optional advanced configurations can be added to config_kwargs
            
        config = types.GenerateContentConfig(**config_kwargs)
        
        logger.debug(f"Calling Gemini with {len(contents)} contents blocks; tools attached: {bool(genai_tools)}")

        import tenacity
        from google.genai.errors import APIError

        def _is_retryable(e: Exception) -> bool:
            return isinstance(e, APIError) and getattr(e, "code", getattr(e, "status_code", 0)) in (429, 503)

        try:
            async for attempt in tenacity.AsyncRetrying(
                wait=tenacity.wait_exponential(
                    multiplier=self.retry_config.get("base_delay", 2),
                    max=self.retry_config.get("max_delay", 15)
                ),
                stop=tenacity.stop_after_attempt(self.retry_config.get("max_retries", 3)),
                retry=tenacity.retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self.client.aio.models.generate_content(
                        model=resolved_model,
                        contents=contents,
                        config=config,
                    )
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

        tool_requests = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, "function_call", None):
                    tc = part.function_call
                    call_id = getattr(tc, "id", None) or ("call_" + str(uuid.uuid4().hex)[:9])
                    
                    thought_sig = getattr(part, "thought_signature", None)
                    if thought_sig:
                        import base64
                        if isinstance(thought_sig, bytes):
                            b64_sig = base64.b64encode(thought_sig).decode("utf-8")
                        else:
                            b64_sig = base64.b64encode(str(thought_sig).encode("utf-8")).decode("utf-8")
                        call_id = f"{call_id}::{b64_sig}"
                        
                    args = tc.args if isinstance(tc.args, dict) else dict(tc.args)
                    tool_requests.append(ToolCallRequest(
                        id=call_id,
                        name=tc.name,
                        arguments=args
                    ))
                
        finish_reason = "stop"
        if tool_requests:
            finish_reason = "tool_calls"
            
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        text_content = None
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            texts = [p.text for p in response.candidates[0].content.parts if p.text]
            if texts:
                text_content = "".join(texts)

        return LLMResponse(
            content=text_content,
            tool_calls=tool_requests,
            finish_reason=finish_reason,
            usage=usage
        )

    def _dict_to_schema(self, schema_dict: dict[str, Any]) -> types.Schema:
        """Recursively translate OpenAI JSON schema dict to google.genai.types.Schema."""
        type_str = schema_dict.get("type", "object").upper()
        if type_str == "FLOAT": 
            type_str = "NUMBER"
        
        properties = None
        if "properties" in schema_dict:
            properties = {k: self._dict_to_schema(v) for k, v in schema_dict["properties"].items()}
            
        items = None
        if "items" in schema_dict:
            items = self._dict_to_schema(schema_dict["items"])
            
        return types.Schema(
            type=type_str,
            description=schema_dict.get("description"),
            properties=properties,
            required=schema_dict.get("required"),
            items=items,
            enum=schema_dict.get("enum")
        )

    def _translate_tools(self, tools: list[dict[str, Any]] | None) -> list[types.Tool] | None:
        if not tools:
            return None
            
        declarations = []
        for tool_dict in tools:
            if tool_dict.get("type") == "function":
                fn = tool_dict["function"]
                params = fn.get("parameters", {})
                
                # Some tools might omit parameters entirely, but genai Schema expects at least empty OBJECT
                if not params:
                    params = {"type": "object", "properties": {}}
                    
                decl = types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=self._dict_to_schema(params)
                )
                declarations.append(decl)
                
        if declarations:
            return [types.Tool(function_declarations=declarations)]
        return None

    def _translate_messages(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> tuple[str | None, list[types.Content]]:
        system_instruction = None
        contents = []
        
        valid_tool_names = set()
        if tools:
            for t in tools:
                if t.get("type") == "function":
                    valid_tool_names.add(t["function"]["name"])
        
        for msg in messages:
            role = msg.get("role")
            content_data = msg.get("content")
            
            if role == "system":
                if isinstance(content_data, str):
                    if system_instruction:
                        system_instruction += "\n\n" + content_data
                    else:
                        system_instruction = content_data
                continue
                
            genai_role = "user"
            if role == "assistant":
                genai_role = "model"
            
            parts = []
            
            # Tools are expected from "assistant" as "functionCall"
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                            
                    call_id = tc.get("id", "")
                    thought_sig = None
                    if "::" in call_id:
                        call_id, thought_sig_b64 = call_id.split("::", 1)
                        import base64
                        try:
                            thought_sig = base64.b64decode(thought_sig_b64)
                        except Exception:
                            pass
                        
                    tc_name = tc["function"]["name"]
                    if tc_name in valid_tool_names:
                        fc = types.FunctionCall(name=tc_name, args=args, id=call_id)
                        part = types.Part(function_call=fc)
                        if thought_sig:
                            part.thought_signature = thought_sig
                    else:
                        text = f"Action: Called tool '{tc_name}' with arguments: {json.dumps(args)}"
                        part = types.Part.from_text(text=text)
                    parts.append(part)
                    
            # Handle "tool" result messages
            if role == "tool":
                result_data = content_data
                tc_id = msg.get("tool_call_id", "")
                if "::" in tc_id:
                    tc_id, _ = tc_id.split("::", 1)

                
                if not isinstance(result_data, dict):
                    result_data = {"result": result_data}
                    
                tc_name = msg.get("name", "unknown")
                genai_role = "user"
                if tc_name in valid_tool_names:
                    resp = types.FunctionResponse(name=tc_name, response=result_data, id=tc_id)
                    part = types.Part(function_response=resp)
                else:
                    text = f"Tool result for '{tc_name}': {json.dumps(result_data)}"
                    part = types.Part.from_text(text=text)
                
                parts.append(part)
                contents.append(types.Content(role=genai_role, parts=parts))
                continue
                
            # Handle Text / Images
            if isinstance(content_data, str):
                if content_data:
                    parts.append(types.Part.from_text(text=content_data))
            elif isinstance(content_data, list):
                for item in content_data:
                    if item.get("type") == "text":
                        parts.append(types.Part.from_text(text=item["text"]))
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            mime_type, b64_data = url[5:].split(";", 1)
                            if b64_data.startswith("base64,"):
                                b64_data = b64_data[7:]
                            import base64
                            image_bytes = base64.b64decode(b64_data)
                            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            
            if role == "assistant" and content_data is None and not parts:
                continue

            if parts:
                contents.append(types.Content(role=genai_role, parts=parts))

        return system_instruction, contents
