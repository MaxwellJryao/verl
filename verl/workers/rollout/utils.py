# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging

import numpy as np
import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


class _UvicornServerAutoPort(uvicorn.Server):
    """Uvicorn Server that reports the system-assigned port when port=0."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.actual_port: int | None = None
        self._startup_done: asyncio.Event = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                sock = self.servers[0].sockets[0]
                self.actual_port = sock.getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._startup_done.set()

    async def get_port(self) -> int | None:
        await self._startup_done.wait()
        return self.actual_port


async def run_uvicorn(app: FastAPI, server_args, server_address) -> tuple[int, asyncio.Task]:
    app.server_args = server_args
    config = uvicorn.Config(app, host=server_address, port=0, log_level="warning")
    server = _UvicornServerAutoPort(config)
    server_task = asyncio.create_task(server.serve())
    server_port = await server.get_port()
    if server_port is None:
        # server.startup() failed. await the task to re-raise exception from server.serve()
        await server_task

        # Fails on unexpected situation.
        raise RuntimeError("Unexpected: HTTP server started without reporting listened port")
    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


def qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive visual placeholder tokens for the Qwen-VL family.

    Some backends rebuild <|image_pad|> and <|video_pad|> runs from raw
    multimodal inputs, so prompts that were already expanded by the HF
    processor need to be collapsed back to a single placeholder per asset.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is None:
        return prompt_ids

    processor_name = processor.__class__.__name__
    image_processor = getattr(processor, "image_processor", None)
    image_processor_name = image_processor.__class__.__name__ if image_processor is not None else ""
    is_qwen_vl_processor = processor_name in {"Qwen2_5_VLProcessor", "Qwen3VLProcessor"} or (
        "Qwen2VLImageProcessor" in image_processor_name
    )
    if not is_qwen_vl_processor:
        return prompt_ids

    token_ids = [token_id for token_id in (getattr(processor, "image_token_id", None), getattr(processor, "video_token_id", None)) if token_id is not None]
    if not token_ids:
        return prompt_ids

    prompt_ids_array = np.asarray(prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else prompt_ids)
    if prompt_ids_array.ndim != 1 or prompt_ids_array.size < 2:
        return prompt_ids

    is_value = np.isin(prompt_ids_array, token_ids)
    if not np.any(is_value[1:] & is_value[:-1]):
        return prompt_ids

    mask = np.ones(len(prompt_ids_array), dtype=bool)
    mask[1:] &= ~(is_value[1:] & is_value[:-1])
    return prompt_ids_array[mask].tolist()
