# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the complete launch/wait/discover/measure path without an NPU."""

import fcntl
from importlib import import_module
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml

run_training = import_module("flagscale_agent.skills.train-run.scripts.training_trial").run_training


@pytest.fixture
def trial(tmp_path, monkeypatch):
    cwd = tmp_path / "checkout"
    cwd.mkdir()
    config = tmp_path / "recipes" / "a nonstandard name.yml"
    config.parent.mkdir()
    recipe = {
        "experiment": {"exp_dir": str(tmp_path / "new attempt"),
                       "task": {"type": "train", "backend": "megatron"},
                       "runner": {"nnodes": 1, "nproc_per_node": 2, "type": "ssh",
                                  "backend": "torchrun", "hostfile": None}},
        "train": {"system": {"logging": {"log_interval": 1}},
                  "model": {"train_iters": 3, "global_batch_size": 4, "seq_length": 8}},
    }
    config.write_text(yaml.safe_dump(recipe))
    executable = tmp_path / "bin" / "flagscale"
    executable.parent.mkdir()
    executable.write_text(f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys, time, yaml
config = pathlib.Path(sys.argv[sys.argv.index('-c') + 1])
recipe = yaml.safe_load(config.read_text())
root = pathlib.Path(recipe['experiment']['exp_dir'])
(root / 'received.json').write_text(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}))
mode = os.environ.get('FAKE_MODE', 'normal')
attempt = root / 'logs/details/host_0_localhost/stamp/default_id/attempt_0'
for rank in range(1 if mode == 'missing_rank' else 2):
    directory = attempt / str(rank)
    directory.mkdir(parents=True)
    text = ''
    if rank == 0:
        for i in range(1, 3 if mode == 'incomplete' else 4):
            text += (f'iteration {i}/3 | elapsed time per iteration (ms): 100 | '
                     'global batch size: 4 | lm loss: 1 | '
                     'number of skipped iterations: 0 | number of nan iterations: 0\n')
    (directory / 'stdout.log').write_text(text.replace('\\n', '\n'))
    (directory / 'stderr.log').write_text('')
if mode == 'extra_attempt':
    (attempt.parent / 'attempt_1').mkdir()
if mode == 'foreign_log':
    external = config.parent / 'external.log'
    external.write_text((attempt / '0/stdout.log').read_text())
    (attempt / '0/stdout.log').unlink()
    (attempt / '0/stdout.log').symlink_to(external)
if mode == 'late_error':
    (attempt / '1/stderr.log').write_text('Traceback (most recent call last):\nfinalization failed')
if mode == 'launcher_error':
    print('[ERROR] Worker exit was masked by sync')
if mode == 'orphan':
    if os.fork() == 0:
        time.sleep(30)
        os._exit(0)
    sys.exit(0)
if mode == 'slow':
    time.sleep(30)
else:
    time.sleep(0.15)
(root / 'finished').write_text('real foreground completion')
sys.exit(7 if mode == 'nonzero' else 0)
''')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    request = {"config_path": str(config), "cwd": str(cwd), "model": "gpt",
               "timeout_seconds": 10, "expected_ranks": 2, "run_id": "b1", "role": "baseline",
               "first_iteration": 1, "end_iteration": 3, "warmup_steps": 1,
               "global_batch_size": 4, "sequence_length": 8}
    return request, recipe, config, Path(recipe["experiment"]["exp_dir"])


def test_foreground_wait_argv_paths_and_measurement(trial):
    request, recipe, config, root = trial
    original = config.read_bytes()
    started = time.monotonic()
    result = run_training(request)
    assert result["status"] == "measured", result
    assert time.monotonic() - started >= 0.15
    assert (root / "finished").exists()
    received = json.loads((root / "received.json").read_text())
    assert received == {"argv": ["train", "gpt", "-c", str(config), "--test"], "cwd": request["cwd"]}
    assert config.read_bytes() == original == (root / "agent_run/config.yaml").read_bytes()
    assert result["mean_step_ms"] == 100
    assert result["tokens_per_second"] == 320
    assert result["launcher_exit_code"] == 0
    assert result["all_rank_completion"] == "not_verified"
    assert Path(result["exit_code_path"]).read_text() == "0\n"
    assert json.loads(Path(result["manifest_path"]).read_text()) == result
    measurement = json.loads(Path(result["measurement_path"]).read_text())
    assert measurement["runs"][0]["completion"]["all_rank_completion"] == "not_verified"


@pytest.mark.parametrize("mode,error", [
    ("nonzero", "code 7"), ("missing_rank", "Rank log"),
    ("incomplete", "missing_iterations"), ("extra_attempt", "exactly one"),
    ("foreign_log", "foreign rank log"), ("late_error", "Fatal error marker"),
    ("launcher_error", "Fatal error marker"), ("orphan", "owned group"),
])
def test_failed_attempt_cannot_be_reported_as_measured(trial, monkeypatch, mode, error):
    request, _, _, _ = trial
    monkeypatch.setenv("FAKE_MODE", mode)
    result = run_training(request)
    assert result["status"] == "failed", result
    assert error in result["error"]
    assert "mean_step_ms" not in result
    assert Path(result["manifest_path"]).exists()
    assert result["launcher_exit_code"] == (7 if mode == "nonzero" else 0)


def test_timeout_stops_only_owned_group_and_preserves_nonzero_exit(trial, monkeypatch):
    request, _, _, root = trial
    monkeypatch.setenv("FAKE_MODE", "slow")
    request["timeout_seconds"] = 0.3
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        result = run_training(request)
        assert result["status"] == "timeout"
        assert result["launcher_exit_code"] < 0
        assert int(Path(result["exit_code_path"]).read_text()) < 0
        assert not (root / "finished").exists()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("change", [
    {"timeout_seconds": 0}, {"timeout_seconds": float("nan")}, {"expected_ranks": 3},
    {"global_batch_size": 8}, {"sequence_length": 16}, {"end_iteration": 2},
    {"warmup_steps": 3}, {"role": "other"}, {"model": "--dryrun"},
])
def test_invalid_request_never_launches(trial, change):
    request, _, _, root = trial
    request.update(change)
    with pytest.raises(ValueError):
        run_training(request)
    assert not root.exists()


@pytest.mark.parametrize("section,key,value", [
    ("runner", "nnodes", 2), ("runner", "backend", "custom"),
    ("runner", "hostfile", "/tmp/hosts"), ("runner", "no_shared_fs", True),
    ("system", "no_shared_fs", True), ("logging", "log_dir", "/tmp/foreign"),
    ("logging", "log_interval", 2), ("experiment", "exp_dir", "/tmp/${now:%Y}"),
])
def test_unsupported_recipe_rejected_before_launch(trial, section, key, value):
    request, recipe, config, root = trial
    target = {"runner": recipe["experiment"]["runner"], "experiment": recipe["experiment"],
              "system": recipe["train"]["system"], "logging": recipe["train"]["system"]["logging"]}[section]
    target[key] = value
    config.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError):
        run_training(request)
    assert not root.exists()


def test_existing_directory_and_concurrent_helper_are_rejected(trial):
    request, _, _, root = trial
    digest = hashlib.sha256(str(Path(request["cwd"]).resolve()).encode()).hexdigest()[:16]
    with (Path(tempfile.gettempdir()) / f"flagscale-training-{os.getuid()}-{digest}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another training_run"):
            run_training(request)
    assert not root.exists()
    root.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        run_training(request)


def test_composed_hydra_config_is_rejected(trial):
    request, recipe, config, root = trial
    recipe["defaults"] = ["other_recipe", "_self_"]
    config.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="composed Hydra"):
        run_training(request)
    assert not root.exists()


def test_cli_stdout_is_compact_and_errors_return_nonzero(trial, monkeypatch, tmp_path):
    request, _, _, _ = trial
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setenv("FAKE_MODE", "late_error")
    completed = subprocess.run([sys.executable, "-m", "flagscale_agent.skills.train-run.scripts.training_trial", "--request", str(path)],
                               capture_output=True, text=True)
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert result["status"] == "failed"
    assert "launcher_log" in result
    assert "argv" not in result
    assert len(completed.stdout) < 2000


def test_sigterm_marks_interrupted_and_stops_owned_launcher(trial, monkeypatch, tmp_path):
    request, _, _, root = trial
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setenv("FAKE_MODE", "slow")
    process = subprocess.Popen([sys.executable, "-m", "flagscale_agent.skills.train-run.scripts.training_trial", "--request", str(path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (root / "received.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (root / "received.json").exists()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 1, stderr
        result = json.loads(stdout)
        assert result["status"] == "interrupted"
        assert result["launcher_exit_code"] < 0
        assert not (root / "finished").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
