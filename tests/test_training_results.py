# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for log-derived performance and quality claims."""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

ANALYSIS_MODULE = "flagscale_agent.skills.train-run.scripts.analyze_training_results"
analysis = import_module(ANALYSIS_MODULE)
analyze_results = analysis.analyze_results


def _cli(tmp_path, request, *args):
    path = tmp_path / "analysis-request.json"
    path.write_text(json.dumps(request))
    # Exercise the documented file entrypoint without an installed Agent package.
    return subprocess.run(
        [sys.executable, "-I", str(Path(analysis.__file__).resolve()),
         "--request", str(path), *args],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )


def _line(i, duration=100, loss=4, *, counters=True, batch=64, grad="1.234"):
    tail = " | number of skipped iterations: 0 | number of nan iterations: 0" if counters else ""
    return (
        f"[default15]: [2026-09-14 21:14:40] iteration {i:8d}/ 40 | consumed samples: {i * 64} "
        f"| elapsed time per iteration (ms): {duration} | global batch size: {batch} "
        f"| lm loss: {loss} | grad norm: {grad}{tail} |\n"
    )


def _run(tmp_path, name, role, *, duration=100, delta=0, count=40, exit_code=None, counters=True):
    path = tmp_path / f"{name}.log"
    path.write_text(f"startup attempt={name}\n" + "".join(
        _line(i, duration if i > 10 else 3000, f"{8 - i / 10 + delta:.6E}", counters=counters)
        for i in range(1, count + 1)
    ))
    result = {"run_id": name, "role": role, "log_path": str(path)}
    if exit_code is not None:
        exit_path = tmp_path / f"{name}.exit"
        exit_path.write_text(str(exit_code) + "\n")
        result["exit_code_path"] = str(exit_path)
    return result


def _analyze(runs, **kwargs):
    return analyze_results(runs=runs, end_iteration=40, global_batch_size=64,
                           sequence_length=512, **kwargs)


def test_real_session_regression_metrics_and_actual_loss_difference(tmp_path):
    baseline = _run(tmp_path, "b", "baseline", duration=135.15)
    candidate = _run(tmp_path, "c", "candidate", duration=57.7566666667)
    path = tmp_path / "c.log"
    lines = path.read_text().splitlines(keepends=True)
    lines[35] = _line(35, 57.7566666667, "4.500535E+00")
    path.write_text("".join(lines))
    result = _analyze([baseline, candidate], loss_atol=1e-6, max_run_variation_pct=5)
    assert result["status"] == "ok"
    assert result["runs"][0]["measurement"]["step_time_ms"]["count"] == 30
    performance = result["comparison"]["performance"]
    assert performance["observed_speedup"] == pytest.approx(2.33998961159, rel=1e-10)
    assert performance["time_reduction_pct"] == pytest.approx(57.264767541)
    assert performance["throughput_improvement_pct"] == pytest.approx(133.998961159, rel=1e-10)
    loss = result["comparison"]["loss_checks"][0]
    assert loss["max_absolute_difference"] == pytest.approx(0.000535)
    assert loss["max_difference_iteration"] == 35
    assert loss["compared_iterations"] == 40
    assert loss["status"] == "outside_tolerance"
    assert result["comparison"]["repeatability"]["status"] == "insufficient_baseline_or_candidate_repeats"
    assert result["comparison"]["acceptance_checks"]["all_launcher_exits_successful"] is False


def test_missing_path_returns_structured_error_and_does_not_create_report(tmp_path):
    proc = _cli(tmp_path, dict(
        runs=[{"run_id": "bad", "role": "baseline", "log_path": str(tmp_path / "missing")}],
        end_iteration=40, global_batch_size=64, sequence_length=512,
        output_path=str(tmp_path / "result.json"),
    ))
    assert proc.returncode == 2
    result = json.loads(proc.stdout)
    assert result["status"] == "error"
    assert "missing" in result["error"]
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("field,value", [("loss", "nan"), ("loss", "-Inf"), ("duration", "NaN"), ("grad", "+infinity")])
def test_nonfinite_values_are_not_silently_skipped_or_serialized_as_nan(tmp_path, field, value):
    b = _run(tmp_path, "b", "baseline")
    c = _run(tmp_path, "c", "candidate")
    path = tmp_path / "c.log"
    lines = path.read_text().splitlines(keepends=True)
    lines[15] = _line(15, **{field: value})
    path.write_text("".join(lines))
    result = _analyze([b, c], loss_atol=0.1)
    assert result["status"] == "invalid_evidence"
    assert result["runs"][1]["measurement"]["nonfinite_metrics"][0]["iteration"] == 15
    assert result["comparison"]["performance"]["observed_speedup"] is None
    assert result["comparison"]["loss_checks"][0]["status"] == "invalid_evidence"
    json.dumps(result, allow_nan=False)


def test_partial_logs_do_not_prove_completion_or_speedup(tmp_path):
    b = _run(tmp_path, "b", "baseline")
    c = _run(tmp_path, "c", "candidate", count=31)
    result = _analyze([b, c])
    run = result["runs"][1]
    assert run["measurement"]["missing_iterations"] == list(range(32, 41))
    assert run["completion"]["iteration_coverage"] == "incomplete"
    assert run["completion"]["launcher_status"] == "unknown"
    assert run["measurement"]["counters"]["nan_iterations"]["total"] is None
    assert result["comparison"]["performance"]["observed_speedup"] is None


def test_complete_log_with_failed_exit_is_not_a_successful_attempt(tmp_path):
    b = _run(tmp_path, "b", "baseline", exit_code=0)
    c = _run(tmp_path, "c", "candidate", exit_code=1)
    result = _analyze([b, c])
    assert result["runs"][1]["completion"]["iteration_coverage"] == "complete"
    assert result["runs"][1]["completion"]["launcher_status"] == "failed"
    assert result["status"] == "invalid_evidence"


def test_complete_iterations_alone_do_not_prove_launcher_or_rank_exit(tmp_path):
    run = _run(tmp_path, "b", "baseline")
    result = _analyze([run])
    assert result["runs"][0]["completion"] == {
        "iteration_coverage": "complete", "launcher_status": "unknown",
        "launcher_exit_code": None, "all_rank_completion": "not_verified", "exit_evidence": None,
    }


def test_duplicate_iteration_invalidates_rank_or_attempt_mixture(tmp_path):
    run = _run(tmp_path, "b", "baseline")
    with (tmp_path / "b.log").open("a") as handle:
        handle.write(_line(15, duration=1))
    result = _analyze([run])
    assert result["status"] == "invalid_evidence"
    assert result["runs"][0]["measurement"]["duplicate_iterations"] == [15]
    assert result["runs"][0]["measurement"]["observed_iteration_count"] == 40


def test_zero_loss_is_preserved_and_relative_tolerance_uses_baseline(tmp_path):
    b = _run(tmp_path, "b", "baseline")
    c = _run(tmp_path, "c", "candidate")
    for name, loss in (("b", "0.0"), ("c", "0.001")):
        path = tmp_path / f"{name}.log"
        lines = path.read_text().splitlines(keepends=True)
        lines[1] = _line(1, loss=loss)
        path.write_text("".join(lines))
    check = _analyze([b, c], loss_rtol=0.1)["comparison"]["loss_checks"][0]
    assert check["baseline_loss_at_max_difference"] == 0
    assert check["status"] == "outside_tolerance"


def test_missing_counters_are_unknown_not_zero(tmp_path):
    run = _run(tmp_path, "b", "baseline", counters=False)
    result = _analyze([run])
    counters = result["runs"][0]["measurement"]["counters"]
    assert counters["skipped_iterations"]["status"] == "unknown"
    assert counters["nan_iterations"]["total"] is None


def test_counters_sum_log_intervals_and_detect_quality_anomaly(tmp_path):
    b = _run(tmp_path, "b", "baseline")
    c = _run(tmp_path, "c", "candidate")
    path = tmp_path / "c.log"
    text = path.read_text().replace("number of skipped iterations: 0", "number of skipped iterations: 1", 2)
    path.write_text(text)
    result = _analyze([b, c], loss_atol=0)
    assert result["runs"][1]["measurement"]["counters"]["skipped_iterations"]["total"] == 2
    assert not result["comparison"]["acceptance_checks"]["all_logged_skip_nan_counters_known_and_zero"]


def test_two_candidates_and_one_baseline_are_not_repeated_baseline_evidence(tmp_path):
    runs = [_run(tmp_path, "b", "baseline", duration=130),
            _run(tmp_path, "c1", "candidate", duration=55),
            _run(tmp_path, "c2", "candidate", duration=56)]
    result = _analyze(runs, max_run_variation_pct=10)
    assert result["comparison"]["repeatability"]["status"] == "insufficient_baseline_or_candidate_repeats"
    assert not result["comparison"]["acceptance_checks"]["repeatability_within_declared_limit"]


def test_balanced_interleaved_pairs_use_each_baseline_and_declared_limits(tmp_path):
    runs = [_run(tmp_path, "b1", "baseline", duration=130, exit_code=0),
            _run(tmp_path, "c1", "candidate", duration=55, exit_code=0),
            _run(tmp_path, "c2", "candidate", duration=56, exit_code=0, delta=0.001),
            _run(tmp_path, "b2", "baseline", duration=131, exit_code=0, delta=0.001)]
    result = _analyze(runs, loss_atol=0, max_run_variation_pct=5)
    comparison = result["comparison"]
    assert comparison["repeatability"]["status"] == "within_declared_variation_limit"
    assert [c["baseline_run_id"] for c in comparison["loss_checks"]] == ["b1", "b2"]
    assert all(c["max_absolute_difference"] == 0 for c in comparison["loss_checks"])
    assert comparison["acceptance_checks"]["all_launcher_exits_successful"]
    assert comparison["acceptance_checks"]["all_rank_completion"] == "not_verified"


@pytest.mark.parametrize("order,status", [
    (["baseline", "baseline", "candidate", "candidate"], "not_interleaved_pairs"),
    (["baseline", "candidate", "baseline", "candidate"], "outside_declared_variation_limit"),
])
def test_nonpaired_or_noisy_repeats_do_not_meet_repeatability_limit(tmp_path, order, status):
    runs = [_run(tmp_path, str(i), role, duration=100 + 30 * i) for i, role in enumerate(order)]
    result = _analyze(runs, max_run_variation_pct=1)
    assert result["comparison"]["repeatability"]["status"] == status


def test_no_implicit_loss_tolerance_or_historical_noise_threshold(tmp_path):
    runs = [_run(tmp_path, str(i), role) for i, role in enumerate(
        ["baseline", "candidate", "baseline", "candidate"])]
    result = _analyze(runs)
    assert result["comparison"]["repeatability"]["status"] == "variation_limit_not_set"
    assert result["comparison"]["loss_checks"][0]["status"] == "tolerance_not_set"


def test_wrong_batch_size_is_detected(tmp_path):
    run = _run(tmp_path, "b", "baseline")
    path = tmp_path / "b.log"
    path.write_text(path.read_text().replace("global batch size: 64", "global batch size: 32"))
    result = _analyze([run])
    assert result["status"] == "invalid_evidence"
    assert result["runs"][0]["measurement"]["batch_size_mismatch_iterations"] == list(range(1, 41))


def test_json_output_is_generated_and_cannot_overwrite_evidence(tmp_path):
    run = _run(tmp_path, "b", "baseline")
    target = tmp_path / "result.json"
    result = _analyze([run], output_path=str(target))
    assert json.loads(target.read_text()) == result
    assert len(result["runs"][0]["log_evidence"]["sha256"]) == 64
    original = (tmp_path / "b.log").read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        _analyze([run], output_path=run["log_path"])
    assert (tmp_path / "b.log").read_bytes() == original


def test_reused_path_cannot_be_counted_as_independent_run(tmp_path):
    run = _run(tmp_path, "b", "baseline")
    other = dict(run, run_id="b2")
    with pytest.raises(ValueError, match="independent"):
        _analyze([run, other])


def test_hard_links_cannot_be_counted_as_independent_runs(tmp_path):
    run = _run(tmp_path, "b", "baseline")
    alias = tmp_path / "linked.log"
    os.link(run["log_path"], alias)
    other = dict(run, run_id="c", role="candidate", log_path=str(alias))
    with pytest.raises(ValueError, match="same physical log file"):
        _analyze([run, other])


def test_copied_identical_logs_cannot_prove_repeatability(tmp_path):
    runs = [_run(tmp_path, str(i), role) for i, role in enumerate(
        ["baseline", "candidate", "baseline", "candidate"])]
    original = (tmp_path / "0.log").read_bytes()
    (tmp_path / "2.log").write_bytes(original)
    target = tmp_path / "results.json"
    with pytest.raises(ValueError, match="byte-identical log evidence"):
        _analyze(runs, loss_atol=0, max_run_variation_pct=5, output_path=str(target))
    assert not target.exists()


@pytest.mark.parametrize("label,value,field", [
    ("grad norm", "NOT_A_NUMBER", "grad_norm"),
    ("lm loss", "broken", "loss"),
    ("elapsed time per iteration (ms)", "57.7oops", "step_ms"),
    ("global batch size", "many", "global_batch_size"),
    ("number of skipped iterations", "unknown", "skipped_iterations"),
    ("number of nan iterations", "", "nan_iterations"),
])
def test_present_malformed_fields_are_invalid_not_absent(tmp_path, label, value, field):
    run = _run(tmp_path, "b", "baseline")
    path = tmp_path / "b.log"
    lines = path.read_text().splitlines(keepends=True)
    import re
    lines[15] = re.sub(re.escape(label) + r": [^|]*", f"{label}: {value} ", lines[15])
    path.write_text("".join(lines))
    result = _analyze([run])
    measurement = result["runs"][0]["measurement"]
    assert result["status"] == "invalid_evidence"
    assert measurement["malformed_metrics"] == [{"iteration": 15, "field": field, "line": 16}]
    if field in ("skipped_iterations", "nan_iterations"):
        assert measurement["counters"][field]["status"] == "invalid"


def test_entirely_malformed_metric_row_is_not_dropped(tmp_path):
    run = _run(tmp_path, "b", "baseline")
    path = tmp_path / "b.log"
    lines = path.read_text().splitlines(keepends=True)
    lines[15] = "iteration 15/40 | elapsed time per iteration (ms): bad | lm loss: bad |\n"
    path.write_text("".join(lines))
    result = _analyze([run])
    measurement = result["runs"][0]["measurement"]
    assert result["status"] == "invalid_evidence"
    assert measurement["observed_iteration_count"] == 40
    assert {item["field"] for item in measurement["malformed_metrics"]} == {"step_ms", "loss"}


def test_omitted_optional_metric_remains_unknown_without_parse_error(tmp_path):
    run = _run(tmp_path, "b", "baseline", counters=False)
    path = tmp_path / "b.log"
    path.write_text(path.read_text().replace(" | grad norm: 1.234", ""))
    result = _analyze([run])
    assert result["status"] == "ok"
    assert result["runs"][0]["measurement"]["malformed_metrics"] == []
    assert result["runs"][0]["measurement"]["counters"]["nan_iterations"]["status"] == "unknown"


@pytest.mark.parametrize("entrypoint", ["file", "module"])
@pytest.mark.parametrize("detail", ["summary", "full"])
def test_cli_with_request_file(tmp_path, entrypoint, detail):
    request = {
        "runs": [_run(tmp_path, "b", "baseline")], "end_iteration": 40,
        "global_batch_size": 64, "sequence_length": 512,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    if entrypoint == "file":
        proc = _cli(tmp_path, request, "--detail", detail)
    else:
        proc = subprocess.run(
            [sys.executable, "-m", ANALYSIS_MODULE, "--request", str(path), "--detail", detail],
            capture_output=True, text=True, check=False,
        )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    if detail == "full":
        assert result["runs"][0]["measurement"]["step_time_ms"]["count"] == 30
    else:
        assert result["detail"] == "summary"
        assert result["runs"][0]["mean_step_time_ms"] == 100


def test_cli_summary_uses_same_aggregate_for_timing_throughput_and_speedup(tmp_path):
    runs = [_run(tmp_path, "b1", "baseline", duration=130, exit_code=0),
            _run(tmp_path, "c1", "candidate", duration=50, exit_code=0),
            _run(tmp_path, "c2", "candidate", duration=60, exit_code=0),
            _run(tmp_path, "b2", "baseline", duration=150, exit_code=0)]
    target = tmp_path / "report.json"
    proc = _cli(tmp_path, dict(
        runs=runs, end_iteration=40, global_batch_size=64, sequence_length=512,
        loss_atol=0, max_run_variation_pct=25, output_path=str(target),
    ))
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    full = json.loads(target.read_text())
    assert summary["detail"] == "summary"
    assert summary["output_path"] == str(target)
    assert "log_evidence" not in summary["runs"][0]
    assert full["runs"][0]["log_evidence"]["path"] == runs[0]["log_path"]
    performance = summary["comparison"]["performance"]
    assert performance["median_run_mean_step_time_ms"] == {"baseline": 140, "candidate": 55}
    assert performance["tokens_per_second"]["baseline"] == pytest.approx(64 * 512 * 1000 / 140)
    assert performance["tokens_per_second"]["candidate"] == pytest.approx(64 * 512 * 1000 / 55)
    assert performance["observed_speedup"] == pytest.approx(140 / 55)
    assert performance["run_variation_pct"]["baseline"] == pytest.approx(100 * 20 / 140)
    assert summary["comparison"]["acceptance_checks"] == full["comparison"]["acceptance_checks"]
    assert summary["comparison"]["acceptance_checks"]["all_rank_completion"] == "not_verified"


def test_cli_summary_preserves_unknown_exits_counters_and_unset_quality_limits(tmp_path):
    runs = [_run(tmp_path, "b", "baseline", counters=False),
            _run(tmp_path, "c", "candidate", duration=50)]
    proc = _cli(tmp_path, dict(
        runs=runs, end_iteration=40, global_batch_size=64, sequence_length=512,
    ))
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["status"] == "ok"  # Evidence validity is not acceptance.
    assert summary["output_path"] is None
    assert summary["runs"][0]["launcher_status"] == "unknown"
    assert summary["runs"][0]["skip_nan_counters"]["nan_iterations"] == "unknown"
    assert summary["comparison"]["loss_checks"][0]["status"] == "tolerance_not_set"
    checks = summary["comparison"]["acceptance_checks"]
    assert checks["all_launcher_exits_successful"] is False
    assert checks["all_logged_skip_nan_counters_known_and_zero"] is False
    assert checks["all_loss_checks_within_declared_tolerance"] is False
    assert checks["repeatability_within_declared_limit"] is False


def test_cli_summary_keeps_invalid_evidence_errors_without_reported_speedup(tmp_path):
    runs = [_run(tmp_path, "b", "baseline"),
            _run(tmp_path, "c", "candidate", duration=1, count=20, exit_code=1)]
    proc = _cli(tmp_path, dict(
        runs=runs, end_iteration=40, global_batch_size=64, sequence_length=512,
    ))
    assert proc.returncode == 1
    summary = json.loads(proc.stdout)
    assert summary["status"] == "invalid_evidence"
    assert summary["runs"][1]["iteration_coverage"] == "incomplete"
    assert summary["runs"][1]["launcher_status"] == "failed"
    assert set(summary["runs"][1]["errors"]) == {"missing_iterations", "launcher_failed"}
    assert summary["runs"][1]["mean_step_time_ms"] is None
    assert summary["runs"][1]["tokens_per_second"] is None
    performance = summary["comparison"]["performance"]
    assert performance["observed_speedup"] is None
    assert performance["tokens_per_second"] == {"baseline": None, "candidate": None}


def test_cli_full_detail_matches_saved_full_report_and_summary_reduces_context(tmp_path):
    runs = [_run(tmp_path, str(i), role, duration=100 + i) for i, role in enumerate(
        ["baseline", "candidate"] * 4)]
    request = dict(runs=runs, end_iteration=40, global_batch_size=64, sequence_length=512,
                   loss_atol=0, max_run_variation_pct=20,
                   output_path=str(tmp_path / "report.json"))
    compact = _cli(tmp_path, request)
    full = _cli(tmp_path, request, "--detail", "full")
    assert compact.returncode == full.returncode == 0
    assert json.loads(full.stdout) == json.loads((tmp_path / "report.json").read_text())
    assert "log_evidence" in json.loads(full.stdout)["runs"][0]
    assert len(compact.stdout) < len(full.stdout) / 2
    invalid = _cli(tmp_path, dict(request, output_path=str(tmp_path / "invalid.json")),
                   "--detail", "verbose")
    assert invalid.returncode == 2
    assert "invalid choice" in invalid.stderr
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("overrides", [
    {"warmup_steps": 40}, {"warmup_steps": -1}, {"first_iteration": True},
    {"loss_atol": float("nan")}, {"loss_rtol": -0.1}, {"max_run_variation_pct": float("inf")},
])
def test_invalid_analysis_contract_is_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        _analyze([_run(tmp_path, "b", "baseline")], **overrides)
