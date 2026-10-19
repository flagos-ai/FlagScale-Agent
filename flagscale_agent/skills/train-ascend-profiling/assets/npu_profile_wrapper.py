#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone entrypoint generated for bounded torch_npu training profiling."""

import functools
import importlib
import inspect
import json
import os
import runpy
import socket
import sys
import time
from pathlib import Path

CONFIG = None  # populated by generator


def _function_file(function):
    return inspect.getsourcefile(inspect.unwrap(function))


def _is_failure(error):
    return error is not None and not (
        isinstance(error, SystemExit) and error.code in (None, 0)
    )


def _check_kwargs(function, values):
    """Check the installed API, without dropping requested options."""
    signature = inspect.signature(function)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        missing = set(values) - set(signature.parameters)
        if missing:
            raise RuntimeError(f"Installed profiler API does not accept: {sorted(missing)}")


class _ProfileSession:
    def __init__(self, config, training, torch, rank, world_size):
        self.config = config
        self.torch = torch
        self.profiler = None
        self.calls = 0
        self.callbacks = 0
        self.window_finished = False
        self.required = config["wait"] + config["warmup"] + config["active"]
        self.directory = Path(config["profile_output"]) / f"rank-{rank:05d}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.status = {
            "schema_version": 1,
            "status": "starting",
            "entrypoint_status": "running",
            "rank": rank,
            "world_size": world_size,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "entrypoint": config["entrypoint"],
            "training_module_file": training.__file__,
            "train_function_file": _function_file(training.train),
            "train_step_function_file": _function_file(training.train_step),
            "config": config,
            "started_at": time.time(),
            "step_unit": "normally returned train_step calls; not necessarily successful optimizer updates",
            "required_train_step_calls": self.required,
            "active_call_range_1based": [config["wait"] + config["warmup"] + 1, self.required],
            "calls_in_schedule": [],
            "limitations": [
                "Call indices are relative to this train invocation, not absolute checkpoint iterations.",
                "Profiler intervals may include inter-step logging, evaluation, checkpoint or scheduling work.",
                "Window completion and callback delivery do not prove complete NPU events or numerical correctness.",
            ],
        }

    def write_status(self):
        self.status.update(
            returned_train_step_calls=self.calls,
            trace_callbacks=self.callbacks,
            window_complete=self.window_finished,
        )
        target = self.directory / "wrapper-status.json"
        temporary = self.directory / "wrapper-status.json.tmp"
        temporary.write_text(json.dumps(self.status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def start(self):
        npu_profiler = importlib.import_module("torch_npu.profiler")
        self.status["profiler_module_file"] = getattr(npu_profiler, "__file__", None)
        handler = npu_profiler.tensorboard_trace_handler(
            str(self.directory / "trace"), worker_name=f"rank{self.status['rank']}"
        )

        def on_trace_ready(profiler):
            handler(profiler)
            self.callbacks += 1

        options = {
            "activities": [npu_profiler.ProfilerActivity.CPU, npu_profiler.ProfilerActivity.NPU],
            "schedule": npu_profiler.schedule(
                wait=self.config["wait"], warmup=self.config["warmup"],
                active=self.config["active"], repeat=1,
            ),
            "on_trace_ready": on_trace_ready,
            "record_shapes": self.config["record_shapes"],
            "with_stack": self.config["with_stack"],
            "profile_memory": self.config["profile_memory"],
            "experimental_config": npu_profiler.experimental_config._ExperimentalConfig(
                profiler_level=getattr(npu_profiler.ProfilerLevel, self.config["level"])
            ),
        }
        _check_kwargs(npu_profiler.profile, options)
        self.profiler = npu_profiler.profile(**options)
        self.profiler.start()
        self.status["status"] = "collecting"
        self.write_status()
        print(f"[NPU wrapper] rank={self.status['rank']} output={self.directory}", flush=True)

    def returned_step(self, iteration):
        self.calls += 1
        if self.profiler is None:
            return
        self.status["calls_in_schedule"].append({
            "call": self.calls,
            "iteration_argument": iteration if isinstance(iteration, int) else None,
        })
        self.profiler.step()
        if self.calls >= self.required:
            self.stop()
            self.window_finished = True
            self.status["status"] = "window_complete_pending_training_end"
            self.write_status()

    def stop(self):
        # Clear first so a failing stop is never retried by a second cleanup path.
        profiler, self.profiler = self.profiler, None
        if profiler is not None:
            profiler.stop()

    def finish(self, training_error):
        cleanup_error = None
        try:
            self.stop()
        except BaseException as exc:
            cleanup_error = exc
            self.status["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(training_error, SystemExit):
            self.status["training_exit_code"] = training_error.code
        if _is_failure(training_error):
            self.status["status"] = "failed"
            self.status["error"] = {"type": type(training_error).__name__, "message": str(training_error)}
        elif cleanup_error is not None:
            self.status["status"] = "failed"
        elif self.window_finished and self.callbacks:
            self.status["status"] = "window_complete_needs_artifact_validation"
        else:
            self.status["status"] = "incomplete"
        self.status["ended_at"] = time.time()
        try:
            self.write_status()
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            if not _is_failure(training_error):
                raise cleanup_error
            print(f"[NPU wrapper] Cleanup also failed: {cleanup_error}", file=sys.stderr, flush=True)
        if self.status["status"] == "incomplete":
            print("[NPU wrapper] Incomplete window or missing trace callback; inspect wrapper-status.json.",
                  file=sys.stderr, flush=True)


    def finish_entrypoint(self, error):
        # pretrain may still save or evaluate after train has returned.
        self.status["entrypoint_status"] = "failed" if _is_failure(error) else "completed"
        if isinstance(error, SystemExit):
            self.status["entrypoint_exit_code"] = error.code
        if _is_failure(error):
            self.status["status"] = "failed"
            self.status["entrypoint_error"] = {"type": type(error).__name__, "message": str(error)}
        self.status["entrypoint_ended_at"] = time.time()
        try:
            self.write_status()
        except BaseException as exc:
            if not _is_failure(error):
                raise
            print(f"[NPU wrapper] Final status write also failed: {exc}", file=sys.stderr, flush=True)


def run(config, training_args=None):
    """Run the original entrypoint once, temporarily instrumenting its training module."""
    if config is None:
        raise RuntimeError("Generate a wrapper with generate_profile_wrapper.py before running it")
    entrypoint = Path(config["entrypoint"]).resolve()
    if not entrypoint.is_file() or entrypoint == Path(__file__).resolve():
        raise ValueError("Original entrypoint must be a different existing Python file")
    saved_argv, saved_path = sys.argv[:], sys.path[:]
    arguments = list(sys.argv[1:] if training_args is None else training_args)
    sys.argv[:] = [str(entrypoint), *arguments]
    sys.path[:] = [str(entrypoint.parent), *config["python_paths"], *saved_path[1:]]
    training = None
    original_train = original_step = profiled_train = patched_step = None
    state = {"entered": False, "session": None, "last_session": None}
    entrypoint_error = None
    try:
        training = importlib.import_module(config["training_module"])
        expected = config["expected_training_file"]
        if expected and Path(training.__file__).resolve() != Path(expected).resolve():
            raise RuntimeError(f"Wrong training module loaded: {training.__file__}; expected {expected}")
        original_train, original_step = training.train, training.train_step
        train_globals = getattr(inspect.unwrap(original_train), "__globals__", None)
        if train_globals is not training.__dict__ or train_globals.get("train_step") is not original_step:
            raise RuntimeError("train does not resolve train_step through the selected module; inspect this version")
        torch = importlib.import_module("torch")

        @functools.wraps(original_step)
        def patched_step(*args, **kwargs):
            session = state["session"]
            if session is None or session.profiler is None:
                value = original_step(*args, **kwargs)
            else:
                with torch.profiler.record_function(f"ascend_train_step/{session.calls + 1}"):
                    value = original_step(*args, **kwargs)
            if session is not None:
                session.returned_step(kwargs.get("iteration"))
            return value

        @functools.wraps(original_train)
        def profiled_train(*args, **kwargs):
            if state["entered"]:
                raise RuntimeError("Wrapper supports one train invocation per process; use a fresh job for restarts")
            state["entered"] = True
            if training.train is not profiled_train or train_globals.get("train_step") is not patched_step:
                raise RuntimeError("Training hooks were replaced after wrapper installation")
            train_args = training.get_args()
            if getattr(train_args, "pytorch_profiler_collect_chakra", False):
                raise RuntimeError("Wrapper does not collect Chakra execution traces; use a compatible dedicated path")
            if getattr(train_args, "profile", False):
                raise RuntimeError("Disable the built-in --profile in the profiling recipe before using this wrapper")
            if getattr(train_args, "skip_train", False):
                raise RuntimeError("skip_train is enabled; wrapper requires actual train_step calls")
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Distributed training must be initialized before the train hook")
            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
            if config["ranks"] is not None:
                if any(rank_id >= world_size for rank_id in config["ranks"]):
                    raise ValueError("Requested profiling rank is outside the actual world size")
                if rank not in config["ranks"]:
                    return original_train(*args, **kwargs)
            session = _ProfileSession(config, training, torch, rank, world_size)
            state["session"] = session
            state["last_session"] = session
            error = None
            try:
                session.start()
                return original_train(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                try:
                    session.finish(error)
                finally:
                    state["session"] = None

        training.train, training.train_step = profiled_train, patched_step
        try:
            runpy.run_path(str(entrypoint), run_name="__main__")
        except SystemExit as exc:
            if not _is_failure(exc) and not state["entered"]:
                raise RuntimeError("Original entrypoint exited without entering the patched train") from exc
            raise
        if not state["entered"]:
            raise RuntimeError("Original entrypoint never entered the patched train; no profile was collected")
    except BaseException as exc:
        entrypoint_error = exc
        raise
    finally:
        if training is not None:
            if profiled_train is not None and training.train is profiled_train:
                training.train = original_train
            if patched_step is not None and training.train_step is patched_step:
                training.train_step = original_step
        sys.argv[:], sys.path[:] = saved_argv, saved_path
        if state["last_session"] is not None:
            state["last_session"].finish_entrypoint(entrypoint_error)


if __name__ == "__main__":
    run(CONFIG)
