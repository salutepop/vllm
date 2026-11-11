# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in research observer; events describe engine actions, not HBM loads."""

import atexit
import contextlib
import dataclasses
import functools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_ROOT = os.environ.get("HS_KV_TRACE_DIR", "")
_FILE = None
_PID = None
_LOCK = threading.Lock()
_COUNTS: dict[tuple[int, str], int] = {}
_STACK = contextlib.ExitStack()


def _close():
    _STACK.close()


atexit.register(_close)


def emit(kind, **fields):
    """Append one event to a process-local buffered JSONL file."""
    global _FILE, _PID, _LOCK
    if not _ROOT:
        return
    pid = os.getpid()
    if pid != _PID:
        _PID = pid
        _LOCK = threading.Lock()
        Path(_ROOT).mkdir(parents=True, exist_ok=True)
        _FILE = _STACK.enter_context(
            open(Path(_ROOT) / f"engine-{pid}.jsonl", "a", buffering=65536)  # noqa: SIM115
        )
    record = {
        "event": kind,
        "pid": pid,
        "wall_ns": time.time_ns(),
        "mono_ns": time.monotonic_ns(),
        **fields,
    }
    with _LOCK:
        _FILE.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
        # Request boundaries make completed requests visible without shutdown.
        if kind in {"request", "free", "cache_layout", "observer_error"} or (
            kind == "output_tokens" and fields.get("finished")
        ):
            _FILE.flush()


def _request(request):
    return {
        "request_id": request.request_id,
        "prompt_tokens": getattr(request, "num_prompt_tokens", None),
        "num_tokens": getattr(request, "num_tokens", None),
        "computed_tokens": getattr(request, "num_computed_tokens", None),
        "output_tokens": getattr(request, "num_output_tokens", None),
    }


def _blocks(blocks):
    if blocks is None:
        return None
    return blocks.get_block_ids() if hasattr(blocks, "get_block_ids") else None


def _layout(config):
    return dataclasses.asdict(config)


def observe(kind):
    """Decorate an engine operation, preserving its return value and exceptions."""

    def decorate(function):
        if not _ROOT:
            return function

        @functools.wraps(function)
        def wrapped(self, *args, **kwargs):
            start = time.monotonic_ns()
            before: Any = None
            request: Any = args[0] if args else kwargs.get("request")
            try:
                if kind == "request":
                    emit(
                        kind,
                        **_request(request),
                        token_ids=list(request.prompt_token_ids or []),
                    )
                elif kind == "free":
                    before = _blocks(self.get_blocks(request.request_id))
                elif kind == "evict":
                    before = {
                        "block_id": request.block_id,
                        "block_hash": str(request.block_hash),
                        "ref_cnt": request.ref_cnt,
                    }
            except Exception as error:
                emit("observer_error", operation=kind, error=repr(error))

            result = function(self, *args, **kwargs)
            try:
                elapsed = time.monotonic_ns() - start
                if kind == "request":
                    return result
                if kind == "scheduler_output":
                    for bundle in result.values():
                        for output in bundle.outputs:
                            if output.new_token_ids or output.finish_reason is not None:
                                emit(
                                    "output_tokens",
                                    request_id=output.request_id,
                                    token_ids=output.new_token_ids,
                                    finished=output.finish_reason is not None,
                                    finish_reason=str(output.finish_reason),
                                    prefill_stats=repr(output.prefill_stats),
                                )
                elif kind == "cache_layout":
                    emit(kind, config=_layout(request))
                elif kind == "manager_init":
                    config = args[0] if args else kwargs["kv_cache_config"]
                    emit(kind, config=_layout(config), max_model_len=self.max_model_len)
                elif kind == "prefix_lookup":
                    emit(
                        kind,
                        **_request(request),
                        hit_tokens=result[1],
                        shared_prefix_boundary=result[2],
                        blocks=_blocks(result[0]),
                        duration_ns=elapsed,
                    )
                elif kind == "allocate":
                    key = (os.getpid(), request.request_id)
                    count = _COUNTS.get(key, 0) + 1
                    _COUNTS[key] = count
                    groups = _blocks(result)
                    fields = {
                        "new_tokens": args[1]
                        if len(args) > 1
                        else kwargs.get("num_new_tokens"),
                        "local_hit_tokens": args[2]
                        if len(args) > 2
                        else kwargs.get("num_new_computed_tokens", 0),
                        "external_hit_tokens": kwargs.get(
                            "num_external_computed_tokens", 0
                        ),
                        "allocation_ok": result is not None,
                        "new_blocks": groups,
                        "pool_free": self.block_pool.get_num_free_blocks(),
                        "duration_ns": elapsed,
                    }
                    if result is not None and (count == 1 or count % 64 == 0):
                        fields["block_table"] = _blocks(
                            self.get_blocks(request.request_id)
                        )
                    emit(kind, **_request(request), **fields)
                elif kind == "free":
                    emit(
                        kind,
                        **_request(request),
                        blocks=before,
                        output_token_ids=list(getattr(request, "output_token_ids", [])),
                        pool_free=self.block_pool.get_num_free_blocks(),
                    )
                    _COUNTS.pop((os.getpid(), request.request_id), None)
                elif kind == "evict" and result:
                    emit(kind, **before)
                elif kind == "pool_allocate":
                    if result:
                        emit(
                            kind,
                            block_ids=[b.block_id for b in result],
                            ref_counts=[b.ref_cnt for b in result],
                            pool_free=self.get_num_free_blocks(),
                        )
                elif kind == "touch":
                    if request:
                        emit(
                            kind,
                            block_ids=[b.block_id for b in request],
                            ref_counts=[b.ref_cnt for b in request],
                        )
                elif kind == "cache_store":
                    blocks = args[1] if len(args) > 1 else kwargs["blocks"]
                    previous = args[2] if len(args) > 2 else kwargs["num_cached_blocks"]
                    full = args[3] if len(args) > 3 else kwargs["num_full_blocks"]
                    group = args[5] if len(args) > 5 else kwargs["kv_cache_group_id"]
                    committed = [
                        [b.block_id, str(b.block_hash), b.ref_cnt]
                        for b in blocks[previous:full]
                        if b.block_hash is not None and not b.is_null
                    ]
                    if committed:
                        emit(
                            kind,
                            request_id=request.request_id,
                            group_id=group,
                            committed=committed,
                        )
            except Exception as error:
                emit("observer_error", operation=kind, error=repr(error))
            return result

        return wrapped

    return decorate


def observe_io(kind):
    """Record filesystem operation requests and completion, preserving failures."""

    def decorate(function):
        if not _ROOT:
            return function

        @functools.wraps(function)
        def wrapped(paths, view, offsets, block_size, use_o_direct=True):
            start = time.monotonic_ns()
            fields = {
                "blocks": len(paths),
                "block_size_bytes": block_size,
                "requested_bytes": len(paths) * block_size,
                "direct_io": use_o_direct,
            }
            try:
                result = function(paths, view, offsets, block_size, use_o_direct)
            except Exception as error:
                emit(
                    kind,
                    **fields,
                    success=False,
                    error=repr(error),
                    duration_ns=time.monotonic_ns() - start,
                )
                raise
            emit(kind, **fields, success=True, duration_ns=time.monotonic_ns() - start)
            return result

        return wrapped

    return decorate
