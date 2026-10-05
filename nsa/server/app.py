"""Ollama-compatible HTTP facade for the full NSA cognitive runtime.

System 1 is a frozen, generation-free local controller. System 2 is a lazy
generative model. NSA owns canonical state, the CCE heartbeat and selective
memory between requests.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from nsa.agent import NSARuntime, RuntimeConfig
from nsa.cognition.system_one_hf import FrozenCausalLMLogitBackend
from nsa.cognition.system_one_runtime import SystemOneController
from nsa.memory.model import MemoryItem, MemoryStore


@dataclass(frozen=True)
class ServerModelSpec:
    alias: str
    hf_id: str
    role: str
    default_max_tokens: int


MODEL_SPECS: dict[str, ServerModelSpec] = {
    "nsa-system1:1.5b": ServerModelSpec(
        "nsa-system1:1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "system1", 32
    ),
    "nsa-system2:3b": ServerModelSpec(
        "nsa-system2:3b", "Qwen/Qwen2.5-3B-Instruct", "system2", 512
    ),
}

ALIASES = {
    "nsa-system1": "nsa-system1:1.5b",
    "nsa-system2": "nsa-system2:3b",
    "nsa:latest": "nsa-system2:3b",
    "nsa": "nsa-system2:3b",
}


def _resolve_alias(name: str) -> str:
    return ALIASES.get(name, name)


class HuggingFaceModelManager:
    """Download-on-demand and cache local causal LMs."""

    def __init__(self, *, device: str = "auto") -> None:
        self.device = device
        self._models: dict[str, tuple[Any, Any]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _lock_for(self, alias: str) -> threading.Lock:
        with self._global_lock:
            return self._locks.setdefault(alias, threading.Lock())

    @staticmethod
    def _torch_device(device: str) -> str:
        if device != "auto":
            return device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self, alias: str) -> tuple[Any, Any]:
        spec = MODEL_SPECS[alias]
        lock = self._lock_for(alias)
        with lock:
            if alias in self._models:
                return self._models[alias]

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Install the server extra: "
                    "pip install 'neural-state-architecture[server]'"
                ) from exc

            device = self._torch_device(self.device)
            dtype = (
                torch.bfloat16
                if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else (torch.float16 if device.startswith("cuda") else torch.float32)
            )

            # Transformers downloads a missing checkpoint automatically and
            # reuses the Hugging Face cache on subsequent starts.
            tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
            model = AutoModelForCausalLM.from_pretrained(
                spec.hf_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            model.eval()
            self._models[alias] = (model, tokenizer)
            return model, tokenizer

    def ensure(self, alias: str) -> None:
        alias = _resolve_alias(alias)
        if alias not in MODEL_SPECS:
            raise ValueError(f"unknown NSA model {alias!r}")
        self._load(alias)

    def get(self, alias: str) -> tuple[Any, Any]:
        alias = _resolve_alias(alias)
        if alias not in MODEL_SPECS:
            raise ValueError(f"unknown NSA model {alias!r}")
        return self._load(alias)

    def loaded(self, alias: str) -> bool:
        return _resolve_alias(alias) in self._models


class NSACognitiveServer:
    """Local single-user cognitive server with persistent System 1."""

    def __init__(
        self,
        *,
        system_one_model: str = "nsa-system1:1.5b",
        system_two_model: str = "nsa-system2:3b",
        device: str = "auto",
        heartbeat_seconds: float = 1.0,
    ) -> None:
        self.system_one_model = _resolve_alias(system_one_model)
        self.system_two_model = _resolve_alias(system_two_model)
        self.heartbeat_seconds = max(0.2, heartbeat_seconds)
        self.models = HuggingFaceModelManager(device=device)
        self.memory = MemoryStore()
        self._memory_lock = threading.Lock()
        self._runtime_lock = threading.RLock()
        self._runtime: NSARuntime | None = None
        self._system_one: SystemOneController | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._system_one_ready = False

    def _load_system_one(self) -> None:
        if self._system_one_ready:
            return
        model, tokenizer = self.models.get(self.system_one_model)
        backend = FrozenCausalLMLogitBackend(model, tokenizer)
        self._system_one = SystemOneController(backend)
        self._runtime = NSARuntime(
            backend=_ServerBackend(self),
            config=RuntimeConfig(
                system_one_enabled=True,
                continuous_enabled=False,
                continuous_interval_seconds=self.heartbeat_seconds,
                history_limit=12,
            ),
            system_one=self._system_one,
        )
        self._system_one_ready = True

    def start(self) -> None:
        self._load_system_one()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._heartbeat, daemon=True, name="nsa-system1"
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _heartbeat(self) -> None:
        while not self._stop.is_set():
            try:
                with self._runtime_lock:
                    if self._runtime is not None:
                        self._runtime.system_one_tick()
            except Exception:
                # System 1 cannot grant authority and must not take down HTTP.
                pass
            self._stop.wait(self.heartbeat_seconds)

    @property
    def runtime(self) -> NSARuntime:
        self.start()
        assert self._runtime is not None
        return self._runtime

    def _memory_policy(self) -> str:
        tick = self.runtime.last_system_one_tick
        return tick.memory_policy if tick else "working"

    def remember(self, content: str, *, role: str, source: str = "chat") -> None:
        policy = self._memory_policy()
        if policy == "discard":
            return
        item = MemoryItem(
            memory_id=str(uuid.uuid4()),
            content=content,
            kind=f"{policy}:{role}",
            provenance_ids=(source,),
        )
        with self._memory_lock:
            self.memory = self.memory.write(item)

    def relevant_memory(self, limit: int = 12) -> list[str]:
        with self._memory_lock:
            active = self.memory.active()
            return [str(item.content) for item in reversed(active[-limit:])]

    def system_one_status(self) -> Mapping[str, Any]:
        runtime = self.runtime
        tick = runtime.last_system_one_tick
        return {
            "ready": self._system_one_ready,
            "model": self.system_one_model,
            "heartbeat_seconds": self.heartbeat_seconds,
            "state": dict(runtime.state.summary()),
            "memory_items": len(self.memory.items),
            "memory_policy": tick.memory_policy if tick else None,
            "salience": tick.salience if tick else None,
            "escalate": tick.escalate if tick else None,
            "decisions": {
                name: {
                    "choice": decision.selected_choice,
                    "confidence": decision.confidence,
                    "uncertainty": decision.uncertainty,
                    "risk": decision.risk,
                    "backend": decision.backend,
                }
                for name, decision in (tick.decisions.items() if tick else ())
            },
        }

    def _build_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        memory = self.relevant_memory()
        lines = []
        if memory:
            lines.append("PERSISTENT NSA MEMORY SELECTED BY SYSTEM 1:")
            lines.extend(f"- {item}" for item in memory)
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            lines.append(f"{role.upper()}: {content}")
        return "\\n".join(lines)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model_alias: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> tuple[str, Mapping[str, Any]]:
        runtime = self.runtime
        for message in messages:
            if message.get("role") in {"user", "system"}:
                self.remember(str(message.get("content", "")), role=str(message.get("role")))

        prompt = self._build_prompt(messages)
        route = self.system_two_model
        if model_alias and model_alias not in {"nsa", "nsa:latest"}:
            route = _resolve_alias(model_alias)
        if route not in MODEL_SPECS:
            route = self.system_two_model

        model, tokenizer = self.models.get(route)
        generated = _generate(
            model, tokenizer, prompt,
            max_tokens or MODEL_SPECS[route].default_max_tokens,
            temperature,
        )
        runtime.observe(generated, source="assistant", confidence=0.8)
        self.remember(generated, role="assistant")

        return generated, {
            "model": route,
            "system_one": self.system_one_status(),
            "memory_items": len(self.memory.items),
        }


def _generate(model: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float) -> str:
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask"}
    }
    kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, int(max_tokens)),
        "do_sample": temperature > 0,
        "return_dict_in_generate": True,
    }
    if temperature > 0:
        kwargs["temperature"] = max(0.01, float(temperature))
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    new_tokens = output.sequences[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class _ServerBackend:
    model = "nsa-system2:3b"

    def __init__(self, server: NSACognitiveServer) -> None:
        self.server = server

    def generate(self, prompt: str, *, state: Mapping[str, Any] | None = None) -> str:
        model, tokenizer = self.server.models.get(self.server.system_two_model)
        return _generate(model, tokenizer, prompt, 512, 0.7)


def _ollama_model_payload(server: NSACognitiveServer, alias: str) -> dict[str, Any]:
    spec = MODEL_SPECS[alias]
    return {
        "name": alias,
        "model": alias,
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size": 0,
        "digest": "nsa",
        "details": {
            "family": "Qwen",
            "parameter_size": alias.rsplit(":", 1)[-1],
            "format": "safetensors",
            "role": spec.role,
            "loaded": server.models.loaded(alias),
            "huggingface_id": spec.hf_id,
        },
    }


def create_app(server: NSACognitiveServer | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the NSA server. Install the server extra."
        ) from exc

    cognitive = server or NSACognitiveServer(
        system_one_model=os.getenv("NSA_SYSTEM_ONE_MODEL", "nsa-system1:1.5b"),
        system_two_model=os.getenv("NSA_SYSTEM_TWO_MODEL", "nsa-system2:3b"),
        device=os.getenv("NSA_MODEL_DEVICE", "auto"),
        heartbeat_seconds=float(os.getenv("NSA_HEARTBEAT_SECONDS", "1.0")),
    )
    app = FastAPI(title="Neural State Architecture", version="0.6.0")

    @app.on_event("startup")
    def startup() -> None:
        cognitive.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        cognitive.stop()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "nsa": cognitive.system_one_status()}

    @app.get("/api/tags")
    def tags() -> dict[str, Any]:
        return {"models": [_ollama_model_payload(cognitive, alias) for alias in MODEL_SPECS]}

    @app.get("/api/show")
    def show(name: str) -> dict[str, Any]:
        alias = _resolve_alias(name)
        if alias not in MODEL_SPECS:
            raise HTTPException(404, "model not found")
        return _ollama_model_payload(cognitive, alias)

    @app.post("/api/pull")
    async def pull(payload: dict[str, Any]) -> StreamingResponse:
        name = _resolve_alias(str(payload.get("name", "")))
        if name not in MODEL_SPECS:
            raise HTTPException(404, f"unknown NSA model {name!r}")

        def events() -> Iterator[bytes]:
            yield (json.dumps({"status": f"pulling {MODEL_SPECS[name].hf_id}"}) + "\\n").encode()
            try:
                cognitive.models.ensure(name)
                yield (json.dumps({"status": "success", "model": name}) + "\\n").encode()
            except Exception as exc:
                yield (json.dumps({"status": "error", "error": str(exc)}) + "\\n").encode()

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/api/chat")
    async def ollama_chat(payload: dict[str, Any]):
        name = _resolve_alias(str(payload.get("model", "nsa")))
        messages = payload.get("messages") or [
            {"role": "user", "content": payload.get("prompt", "")}
        ]
        stream = bool(payload.get("stream", False))
        options = payload.get("options") or {}
        max_tokens = int(options.get("num_predict", payload.get("max_tokens", 512)))
        temperature = float(options.get("temperature", payload.get("temperature", 0.7)))
        text, meta = await asyncio.to_thread(
            cognitive.chat, messages, model_alias=name,
            max_tokens=max_tokens, temperature=temperature,
        )
        if not stream:
            return {
                "model": name,
                "message": {"role": "assistant", "content": text},
                "done": True,
                "nsa": meta,
            }

        def stream_events() -> Iterator[bytes]:
            for token in text.split(" "):
                yield (
                    json.dumps({
                        "model": name,
                        "message": {"role": "assistant", "content": token + " "},
                        "done": False,
                    }) + "\\n"
                ).encode()
            yield (
                json.dumps({
                    "model": name,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "nsa": meta,
                }) + "\\n"
            ).encode()

        return StreamingResponse(stream_events(), media_type="application/x-ndjson")

    @app.post("/api/generate")
    async def ollama_generate(payload: dict[str, Any]):
        prompt = str(payload.get("prompt", ""))
        name = _resolve_alias(str(payload.get("model", "nsa")))
        stream = bool(payload.get("stream", False))
        options = payload.get("options") or {}
        max_tokens = int(options.get("num_predict", 512))
        temperature = float(options.get("temperature", 0.7))
        text, meta = await asyncio.to_thread(
            cognitive.chat, [{"role": "user", "content": prompt}],
            model_alias=name, max_tokens=max_tokens, temperature=temperature,
        )
        if not stream:
            return {"model": name, "response": text, "done": True, "nsa": meta}

        def stream_events() -> Iterator[bytes]:
            for token in text.split(" "):
                yield (
                    json.dumps({"model": name, "response": token + " ", "done": False})
                    + "\\n"
                ).encode()
            yield (
                json.dumps({"model": name, "response": "", "done": True, "nsa": meta})
                + "\\n"
            ).encode()

        return StreamingResponse(stream_events(), media_type="application/x-ndjson")

    @app.get("/v1/models")
    def openai_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "neural-state-architecture",
                }
                for alias in MODEL_SPECS
            ],
        }

    @app.post("/v1/chat/completions")
    async def openai_chat(payload: dict[str, Any]):
        name = _resolve_alias(str(payload.get("model", "nsa")))
        messages = payload.get("messages") or []
        stream = bool(payload.get("stream", False))
        max_tokens = int(payload.get("max_tokens", 512))
        temperature = float(payload.get("temperature", 0.7))
        text, meta = await asyncio.to_thread(
            cognitive.chat, messages, model_alias=name,
            max_tokens=max_tokens, temperature=temperature,
        )
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        if not stream:
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(text.split()),
                },
                "nsa": meta,
            }

        def sse() -> Iterator[str]:
            for token in text.split(" "):
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token + " "},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\\n\\n"
            yield (
                f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'model': name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\\n\\n"
            )
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/nsa/state")
    def nsa_state() -> Mapping[str, Any]:
        return cognitive.system_one_status()

    @app.post("/nsa/tick")
    async def nsa_tick() -> Mapping[str, Any]:
        await asyncio.to_thread(cognitive.runtime.system_one_tick)
        return cognitive.system_one_status()

    return app


def run_server(host: str = "0.0.0.0", port: int = 11434) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Install the server extra to run the NSA server") from exc
    uvicorn.run(create_app(), host=host, port=port)
