import asyncio
import json
import threading
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..models.cache import get_model_cache
from ..models.error import ErrorDetail, ErrorResponse
from ..utils.gpu import gpu_inference
from ..utils.logger import logger
from ..vlm.utils import detect_multimodal_content, is_vlm_model
from .generator import ChatGenerator
from .schema import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ToolCall,
)

router = APIRouter(prefix="/v1", tags=["chat"])


def _has_multimodal_content(messages) -> bool:
    """Check if messages contain multimodal content (images, audio, video - NOT text)."""
    for msg in messages:
        detection = detect_multimodal_content(msg.content)
        if detection.get("has_images") or detection.get("has_audio") or detection.get("has_video"):
            return True
    return False


def _create_error_response(status_code: int, message: str, error_type: str, code: str, param: str = None):
    """Create a standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=message,
                type=error_type,
                code=code,
                param=param
            )
        ).model_dump()
    )


def _parse_tool_calls(tool_calls_data) -> Optional[List[ToolCall]]:
    """Parse tool calls from generator response."""
    if not tool_calls_data:
        return None
    
    logger.debug(f"Processing {len(tool_calls_data)} tool calls")
    return [
        ToolCall(
            id=tc["id"],
            type=tc["type"],
            function={"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
            index=tc.get("index")
        )
        for tc in tool_calls_data
    ]


def _create_chunk(completion_id: str, created: int, model: str, content: str = "", 
                  reasoning: str = None, tool_calls: List[ToolCall] = None, 
                  finish_reason: str = None) -> str:
    """Create a streaming chunk in SSE format."""
    chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[ChatCompletionChunkChoice(
            index=0,
            delta=ChatMessage(
                role="assistant",
                content=content,
                reasoning=reasoning,
                tool_calls=tool_calls,
            ),
            finish_reason=finish_reason,
        )],
    )
    return f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, http_request: Request):
    try:
        has_multimodal = _has_multimodal_content(request.messages)
        model_is_vlm = is_vlm_model(request.model)
        use_vlm_generator = model_is_vlm
        extra_params = request.get_extra_params()

        if has_multimodal and not model_is_vlm:
            return _create_error_response(
                400,
                "This request contains image/audio/video content, but the selected model "
                "does not appear to be a VLM. Please use a VLM model for multimodal input.",
                "invalid_request_error",
                "model_not_supported_for_multimodal",
                "model",
            )
        
        def progress_callback(processed: int, total: int):
            progress_pct = (processed / total * 100) if total > 0 else 0
            logger.debug(f"Prompt processing progress: {processed}/{total} ({progress_pct:.1f}%)")
        
        # Get appropriate generator (may trigger model download/loading)
        try:
            if use_vlm_generator:
                generator = await asyncio.to_thread(
                    get_model_cache().get_vlm_generator,
                    request.model, extra_params.get("adapter_path")
                )
            else:
                use_cache = request.enable_cache if request.enable_cache is not None else True
                generator = await asyncio.to_thread(
                    get_model_cache().get_generator,
                    request.model, 
                    extra_params.get("adapter_path"),
                    use_cache,
                    request.max_kv_size,
                )
        except Exception as e:
            return _create_error_response(400, str(e), "invalid_request_error", "model_not_found", "model")
        
        # Prepare messages and tools
        messages = [
            {"role": msg.role, "content": msg.content if use_vlm_generator else msg.get_text_content()}
            for msg in request.messages
        ]
        
        tools = [tool.model_dump() for tool in request.tools] if request.tools else None
        if tools:
            logger.debug(f"Request contains {len(tools)} tools: {[t['function']['name'] for t in tools]}")
        
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        
        # Build generation kwargs
        gen_kwargs = {
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
        }
        
        if not use_vlm_generator:
            if tools:
                gen_kwargs["tools"] = tools
            gen_kwargs["use_cache"] = request.enable_cache if request.enable_cache is not None else True
            gen_kwargs["progress_callback"] = progress_callback
        
        # Non-streaming response
        if not request.stream:
            t0 = time.perf_counter()
            async with gpu_inference():
                result = await asyncio.to_thread(generator.generate, **gen_kwargs)
            elapsed = time.perf_counter() - t0
            tool_calls = _parse_tool_calls(result.get("tool_calls"))
            
            prompt_toks = result["prompt_tokens"]
            completion_toks = result["completion_tokens"]
            reasoning_toks = result.get("reasoning_tokens", 0)
            content_toks = completion_toks - reasoning_toks
            ttft = result.get("ttft", 0)
            decode_time = elapsed - ttft
            tps = completion_toks / decode_time if decode_time > 0 else 0
            
            completion_detail = f"(think={reasoning_toks} content={content_toks})" if reasoning_toks else ""
            logger.info(
                f"[{request.model}] prompt={prompt_toks} completion={completion_toks}{completion_detail} "
                f"ttft={ttft:.2f}s decode={decode_time:.2f}s total={elapsed:.2f}s speed={tps:.1f} tok/s"
            )
            
            return JSONResponse(content=ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=result["text"],
                        reasoning=result.get("reasoning"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_toks,
                    completion_tokens=completion_toks,
                    total_tokens=prompt_toks + completion_toks,
                ),
            ).model_dump(exclude_none=True))
        
        # Streaming response
        _SENTINEL = object()

        async def event_generator():
            t0 = time.perf_counter()
            ttft = None
            prompt_toks = completion_toks = reasoning_toks = 0
            cancel_event = threading.Event()

            queue: asyncio.Queue = asyncio.Queue()

            def _produce():
                """Run the sync stream generator in a thread, pushing results to the queue."""
                try:
                    for item in generator.generate_stream(**gen_kwargs):
                        if cancel_event.is_set():
                            break
                        queue.put_nowait(item)
                except Exception as exc:
                    queue.put_nowait(exc)
                finally:
                    queue.put_nowait(_SENTINEL)

            async with gpu_inference():
                loop = asyncio.get_running_loop()
                fut = loop.run_in_executor(None, _produce)

                while True:
                    response = await queue.get()
                    if response is _SENTINEL:
                        break
                    if isinstance(response, Exception):
                        raise response
                    if await http_request.is_disconnected():
                        logger.info(f"[{request.model}] Client disconnected, stopping generation")
                        cancel_event.set()
                        break
                    if ttft is None and (response['text'] or response.get('reasoning')):
                        ttft = time.perf_counter() - t0
                    prompt_toks = response.get('prompt_tokens', prompt_toks)
                    completion_toks = response.get('completion_tokens', completion_toks)
                    reasoning_toks = response.get('reasoning_tokens', reasoning_toks)
                    yield _create_chunk(
                        completion_id, created, request.model,
                        response['text'], response.get('reasoning'),
                        _parse_tool_calls(response.get("tool_calls")),
                        response['finish_reason']
                    )
                    if response['finish_reason']:
                        break

                await fut
            
            elapsed = time.perf_counter() - t0
            decode_time = elapsed - (ttft or 0)
            tps = completion_toks / decode_time if decode_time > 0 else 0
            ttft_str = f"{ttft:.2f}s" if ttft is not None else "N/A"
            content_toks = completion_toks - reasoning_toks
            completion_detail = f"(think={reasoning_toks} content={content_toks})" if reasoning_toks else ""
            logger.info(
                f"[{request.model}] prompt={prompt_toks} completion={completion_toks}{completion_detail} "
                f"ttft={ttft_str} decode={decode_time:.2f}s total={elapsed:.2f}s speed={tps:.1f} tok/s"
            )
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    
    except asyncio.TimeoutError:
        return _create_error_response(
            503,
            "Server is busy. Request timed out waiting for GPU resources.",
            "server_error",
            "timeout",
        )
    except ValueError as e:
        return _create_error_response(400, str(e), "invalid_request_error", "invalid_value")
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {str(e)}", exc_info=True)
        return _create_error_response(
            500, 
            "An unexpected error occurred while processing your request.", 
            "server_error", 
            "internal_error"
        )
