"""
pythia/llm/vllm_backend.py

Async-parallel backend for any OpenAI-compatible server (vLLM, llama.cpp, etc.)
that processes notes in concurrent batches.

The public invoke() method is synchronous — it satisfies BaseLLMBackend and can
be called anywhere in Pythia for single-note classification.

The batch_invoke() method is the parallel entrypoint. The Specialist agent calls
this when classifying a full dataset, firing all notes concurrently up to
max_concurrent at a time.

Memory monitoring is inherited from the original vllm_test_cli.py design and is
optional — pass monitor_memory=True to batch_invoke() to enable it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import psutil

from pythia.llm.base import BaseLLMBackend


class VLLMBackend(BaseLLMBackend):
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11484/v1",
        api_key: str = "not-needed",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_concurrent: int = 10,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.timeout = timeout

    #Public interface (satisfies BaseLLMBackend) 

    def invoke(self, prompt: str) -> str:
        """
        Synchronous single-prompt call.
        Used by Summarizer, Improver, and anywhere a single LLM call is needed.
        Runs the async implementation on a new event loop.
        """
        return asyncio.run(self._async_single(prompt))

    def batch_invoke(
        self,
        prompts:        list[str],
        monitor_memory: bool = False,
    ) -> list[str]:
        """
        Synchronous batched call over a list of prompts.
        Fires all prompts concurrently up to max_concurrent at a time.
        Called by SpecialistAgent.classify_all() instead of looping invoke().
        Returns responses in the same order as the input prompts.
        """
        return asyncio.run(self._async_batch(prompts, monitor_memory))

    #Async implementation
    async def _async_single(self, prompt: str) -> str:
        """Single async request. Shares session setup with the batch path."""
        headers = self._headers()
        async with aiohttp.ClientSession() as session:
            return await self._post(session, prompt, request_id=0, headers=headers)

    async def _async_batch(
        self,
        prompts:        list[str],
        monitor_memory: bool,
    ) -> list[str]:
        """
        Core async batch runner. Semaphore caps concurrency at max_concurrent.
        Results are collected in input order via enumerate + indexed list.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        headers   = self._headers()
        results   = [None] * len(prompts)

        async def bounded(session: aiohttp.ClientSession, prompt: str, idx: int):
            async with semaphore:
                results[idx] = await self._post(
                    session, prompt, request_id=idx, headers=headers
                )

        # Optionally start memory monitor
        stop_event   = None
        monitor_task = None
        if monitor_memory:
            stop_event   = asyncio.Event()
            monitor_task = asyncio.create_task(self._monitor_memory(stop_event))

        async with aiohttp.ClientSession() as session:
            tasks = [
                bounded(session, prompt, i)
                for i, prompt in enumerate(prompts)
            ]
            await asyncio.gather(*tasks, return_exceptions=False)

        if stop_event:
            stop_event.set()
            await monitor_task

        return results

    async def _post(
        self,
        session:    aiohttp.ClientSession,
        prompt:     str,
        request_id: int,
        headers:    dict,
    ) -> str:
        """Single HTTP request to the /chat/completions endpoint."""
        url     = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        start = time.time()
        print(f"[{start:.2f}] Request {request_id} started")

        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            response.raise_for_status()
            data = await response.json()

        elapsed = time.time() - start
        print(f"[{time.time():.2f}] Request {request_id} completed in {elapsed:.2f}s")
        #return data['response'].strip()
        return data["choices"][0]["message"]["content"].strip()

    # Memory monitor (from vllm_test_cli.py)

    async def _monitor_memory(
        self,
        stop_event: asyncio.Event,
        interval:   float = 1.0,
    ) -> None:
        start_time   = time.time()
        peak_memory  = 0.0
        samples: list[float] = []

        print("\n=== Memory Monitoring Started ===")
        print(f"{'Time (s)':<10} {'Used (GB)':<12} {'Available (GB)':<15} {'Percent':<10}")
        print("-" * 50)

        while not stop_event.is_set():
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)
            elapsed = time.time() - start_time
            peak_memory = max(peak_memory, used_gb)
            samples.append(used_gb)

            print(
                f"{elapsed:<10.1f} {used_gb:<12.2f} "
                f"{available_gb:<15.2f} {mem.percent:<10.1f}%"
            )

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                continue

        print("-" * 50)
        print(f"Peak memory:    {peak_memory:.2f} GB")
        print(f"Average memory: {sum(samples) / len(samples):.2f} GB")
        print(f"Memory delta:   {samples[-1] - samples[0]:+.2f} GB")
        print("=== Memory Monitoring Ended ===\n")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }