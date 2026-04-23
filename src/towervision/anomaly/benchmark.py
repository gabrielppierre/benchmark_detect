"""Unified orchestration for the anomaly benchmark."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from towervision.anomaly.benchmark_dataset import materialize_anomaly_benchmark_dataset
from towervision.anomaly.benchmark_reporting import persist_benchmark_report
from towervision.anomaly.benchmark_types import (
    AnomalyBenchmarkDatasetArtifacts,
    AnomalyBenchmarkModelConfig,
    AnomalyBenchmarkTrainingConfig,
    AnomalySeedRunResult,
)
from towervision.utils.io import ensure_dir, read_json, read_yaml, write_json, write_text


def load_benchmark_experiment_config(path: Path) -> dict[str, Any]:
    """Load the anomaly benchmark experiment YAML."""

    payload = read_yaml(path, default={})
    if not isinstance(payload, dict):
        raise ValueError(f"invalid anomaly benchmark config: {path}")
    return payload


def parse_training_config(raw: Mapping[str, Any], ranking: Mapping[str, Any]) -> AnomalyBenchmarkTrainingConfig:
    """Build the shared anomaly training config from YAML."""

    seeds = [int(seed) for seed in raw.get("seeds", [])]
    num_seeds = int(raw.get("num_seeds", len(seeds) or 3))
    if not seeds:
        seeds = [42 + (index * 10) for index in range(num_seeds)]
    iterative_controls = dict(raw.get("iterative_controls", {}))
    return AnomalyBenchmarkTrainingConfig(
        train_only_normal=bool(raw.get("train_only_normal", True)),
        synthetic_in_training=bool(raw.get("synthetic_in_training", False)),
        input_size=int(raw["input_size"]),
        normalization=str(raw["normalization"]),
        feature_extractor=str(raw["feature_extractor"]),
        num_seeds=num_seeds,
        seeds=seeds[:num_seeds],
        operating_recall_floor=float(ranking["operating_recall_floor"]),
        iterative_enabled_for=[str(item) for item in iterative_controls.get("enabled_for", [])],
        max_epochs=int(iterative_controls.get("max_epochs", 100)),
        validate_every=int(iterative_controls.get("validate_every", 1)),
        save_best=bool(iterative_controls.get("save_best", True)),
        save_last=bool(iterative_controls.get("save_last", True)),
        early_stopping=bool(iterative_controls.get("early_stopping", True)),
        monitor=str(iterative_controls.get("monitor", "val_roi_auroc")),
        mode=str(iterative_controls.get("mode", "max")),
        patience=int(iterative_controls.get("patience", 20)),
        min_epochs=int(iterative_controls.get("min_epochs", 25)),
    )


def parse_model_configs(
    repo_root: Path,
    model_paths: Sequence[str],
) -> list[AnomalyBenchmarkModelConfig]:
    """Load per-model configs from YAML files listed in the experiment config."""

    configs: list[AnomalyBenchmarkModelConfig] = []
    for raw_path in model_paths:
        path = _resolve_repo_path(repo_root, raw_path)
        payload = read_yaml(path, default={})
        if not isinstance(payload, dict):
            raise ValueError(f"invalid model config: {path}")
        known_fields = {
            "name",
            "display_name",
            "family",
            "backend",
            "fit_mode",
            "input_size",
            "feature_extractor",
            "notes",
            "extra",
        }
        extra = {key: value for key, value in payload.items() if key not in known_fields}
        extra.update(dict(payload.get("extra", {})))
        configs.append(
            AnomalyBenchmarkModelConfig(
                name=str(payload["name"]),
                display_name=str(payload.get("display_name", payload["name"])),
                family=str(payload["family"]),
                backend=str(payload["backend"]),
                fit_mode=str(payload["fit_mode"]),
                input_size=int(payload.get("input_size", 256)),
                feature_extractor=str(payload.get("feature_extractor", "resnet18")),
                notes=str(payload.get("notes", "")),
                extra=extra,
            )
        )
    return configs


def resolve_benchmark_paths(
    *,
    repo_root: Path,
    params: Mapping[str, Any],
    benchmark_name: str,
) -> dict[str, Path]:
    """Resolve canonical paths for the anomaly benchmark."""

    dataset_name = str(params["dataset"]["name"])
    dataset_version = str(params["dataset"]["version"])
    return {
        "dataset_materialized_dir": (
            repo_root
            / "data"
            / "interim"
            / dataset_name
            / dataset_version
            / "anomaly_benchmark"
            / benchmark_name
            / "dataset"
        ),
        "runs_root": (
            repo_root
            / "runs"
            / "anomaly"
            / dataset_name
            / dataset_version
            / benchmark_name
        ),
        "reports_root": (
            repo_root
            / "reports"
            / "tables"
            / dataset_name
            / dataset_version
            / "anomaly_benchmark"
            / benchmark_name
        ),
    }


def prepare_anomaly_benchmark(
    *,
    repo_root: Path,
    params: Mapping[str, Any],
    benchmark_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare the materialized dataset and roots for the anomaly benchmark."""

    benchmark_name = str(benchmark_config["name"])
    paths = resolve_benchmark_paths(repo_root=repo_root, params=params, benchmark_name=benchmark_name)
    dataset_config = dict(benchmark_config["dataset"])
    artifacts = materialize_anomaly_benchmark_dataset(
        images_manifest_path=_resolve_repo_path(repo_root, params["paths"]["cleaned_images_manifest"]),
        annotations_manifest_path=_resolve_repo_path(repo_root, params["paths"]["cleaned_annotations_manifest"]),
        splits_path=_resolve_repo_path(repo_root, params["paths"]["splits_path"]),
        synthetic_records_path=_resolve_repo_path(repo_root, params["paths"]["synthetic_anomaly_records"]),
        output_dir=paths["dataset_materialized_dir"],
        roi_label=str(dataset_config["roi_label"]),
    )
    ensure_dir(paths["runs_root"])
    ensure_dir(paths["reports_root"])
    return {
        "paths": paths,
        "dataset_artifacts": artifacts,
        "dataset_config": dataset_config,
    }


def build_job_specs(
    *,
    benchmark_config: Mapping[str, Any],
    dataset_artifacts: AnomalyBenchmarkDatasetArtifacts,
    model_configs: Sequence[AnomalyBenchmarkModelConfig],
    training_config: AnomalyBenchmarkTrainingConfig,
    runs_root: Path,
    selected_models: Sequence[str] | None = None,
    selected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Create reproducible job specs for each model and seed."""

    allowed_model_names = set(selected_models) if selected_models else None
    allowed_seeds = set(selected_seeds) if selected_seeds else None
    jobs: list[dict[str, Any]] = []
    for model in model_configs:
        if allowed_model_names is not None and model.name not in allowed_model_names:
            continue
        for seed in training_config.seeds:
            if allowed_seeds is not None and seed not in allowed_seeds:
                continue
            run_dir = runs_root / model.name / f"seed_{seed}"
            jobs.append(
                {
                    "benchmark_name": benchmark_config["name"],
                    "dataset_views": {
                        "root_dir": dataset_artifacts.root_dir.as_posix(),
                        "summary_path": dataset_artifacts.summary_path.as_posix(),
                        "split_manifests": {
                            split_name: path.as_posix()
                            for split_name, path in dataset_artifacts.split_to_manifest_path.items()
                        },
                        "split_root_dirs": {
                            split_name: path.as_posix()
                            for split_name, path in dataset_artifacts.split_to_root_dir.items()
                        },
                        "split_normal_dirs": {
                            split_name: path.as_posix()
                            for split_name, path in dataset_artifacts.split_to_normal_dir.items()
                        },
                        "split_anomaly_dirs": {
                            split_name: path.as_posix()
                            for split_name, path in dataset_artifacts.split_to_anomaly_dir.items()
                        },
                        "crop_padding": dataset_artifacts.crop_padding,
                    },
                    "dataset": dict(benchmark_config["dataset"]),
                    "ranking": dict(benchmark_config["ranking"]),
                    "model": {
                        "name": model.name,
                        "display_name": model.display_name,
                        "family": model.family,
                        "backend": model.backend,
                        "fit_mode": model.fit_mode,
                        "input_size": model.input_size,
                        "feature_extractor": model.feature_extractor,
                        "notes": model.notes,
                        "extra": model.extra,
                    },
                    "training": {
                        "input_size": training_config.input_size,
                        "normalization": training_config.normalization,
                        "feature_extractor": training_config.feature_extractor,
                        "seed": seed,
                        "max_epochs": training_config.max_epochs,
                        "validate_every": training_config.validate_every,
                        "save_best": training_config.save_best,
                        "save_last": training_config.save_last,
                        "early_stopping": training_config.early_stopping,
                        "monitor": training_config.monitor,
                        "mode": training_config.mode,
                        "patience": training_config.patience,
                        "min_epochs": training_config.min_epochs,
                    },
                    "run_dir": run_dir.as_posix(),
                    "seed": seed,
                }
            )
    return jobs


def write_job_specs(job_specs: Sequence[Mapping[str, Any]], *, runs_root: Path) -> Path:
    """Persist the anomaly benchmark job index and job specs."""

    for job in job_specs:
        run_dir = Path(str(job["run_dir"]))
        ensure_dir(run_dir)
        write_json(run_dir / "job.json", dict(job))
    index_path = runs_root / "job_index.json"
    write_json(index_path, list(job_specs))
    return index_path


def collect_benchmark_seed_results(runs_root: Path) -> list[dict[str, Any]]:
    """Collect every structured anomaly result under the benchmark runs root."""

    results_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for result_path in sorted(runs_root.glob("*/seed_*/result.json")):
        result = read_json(result_path, default={})
        if not isinstance(result, dict):
            continue
        model_name = str(result.get("model_name", result_path.parent.parent.name))
        seed = int(result.get("seed", result_path.parent.name.removeprefix("seed_")))
        results_by_key[(model_name, seed)] = result
    return list(results_by_key.values())


def load_completed_result(
    run_dir: Path,
    *,
    expected_backend: str | None = None,
) -> dict[str, Any] | None:
    """Return an existing completed anomaly result for one run directory."""

    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None
    result = read_json(result_path, default={})
    if not isinstance(result, dict):
        return None
    if str(result.get("status")) != "completed":
        return None
    if expected_backend is not None and str(result.get("backend")) != expected_backend:
        return None
    return result


def run_anomaly_benchmark(
    *,
    repo_root: Path,
    params_path: Path,
    benchmark_config_path: Path,
    execute: bool = False,
    selected_models: Sequence[str] | None = None,
    selected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Prepare the anomaly benchmark and optionally execute it."""

    params = read_yaml(params_path)
    benchmark_config = load_benchmark_experiment_config(benchmark_config_path)
    training_config = parse_training_config(
        benchmark_config["training"],
        benchmark_config["ranking"],
    )
    model_configs = parse_model_configs(repo_root, benchmark_config["models"])
    prepared = prepare_anomaly_benchmark(
        repo_root=repo_root,
        params=params,
        benchmark_config=benchmark_config,
    )
    runs_root = prepared["paths"]["runs_root"]
    job_specs = build_job_specs(
        benchmark_config=benchmark_config,
        dataset_artifacts=prepared["dataset_artifacts"],
        model_configs=model_configs,
        training_config=training_config,
        runs_root=runs_root,
        selected_models=selected_models,
        selected_seeds=selected_seeds,
    )
    job_index_path = write_job_specs(job_specs, runs_root=runs_root)

    if execute:
        seed_results = [execute_benchmark_job(job, repo_root=repo_root) for job in job_specs]
        report_seed_results = collect_benchmark_seed_results(runs_root)
    else:
        seed_results = [
            AnomalySeedRunResult(
                model_name=str(job["model"]["name"]),
                display_name=str(job["model"]["display_name"]),
                seed=int(job["seed"]),
                status="planned",
                backend=str(job["model"]["backend"]),
                fit_mode=str(job["model"]["fit_mode"]),
                model_artifact_path=(Path(str(job["run_dir"])) / "model.json").as_posix(),
                train_log_path=(Path(str(job["run_dir"])) / "train.log").as_posix(),
                train_history_path=(Path(str(job["run_dir"])) / "train_history.csv").as_posix(),
                threshold_selection_path=(Path(str(job["run_dir"])) / "threshold_selection.json").as_posix(),
                val_scores_path=(Path(str(job["run_dir"])) / "val_scores.csv").as_posix(),
                test_scores_path=(Path(str(job["run_dir"])) / "test_scores.csv").as_posix(),
                val_metrics_path=(Path(str(job["run_dir"])) / "val_metrics.json").as_posix(),
                test_metrics_path=(Path(str(job["run_dir"])) / "test_metrics.json").as_posix(),
                generator_breakdown_path=(Path(str(job["run_dir"])) / "generator_breakdown.csv").as_posix(),
                anomaly_type_breakdown_path=(Path(str(job["run_dir"])) / "anomaly_type_breakdown.csv").as_posix(),
                severity_breakdown_path=(Path(str(job["run_dir"])) / "severity_breakdown.csv").as_posix(),
                notes=["job planned but not executed"],
            ).to_dict()
            for job in job_specs
        ]
        collected_results = collect_benchmark_seed_results(runs_root)
        report_seed_results = collected_results or seed_results

    report_artifacts = persist_benchmark_report(
        root_dir=prepared["paths"]["reports_root"],
        benchmark_name=str(benchmark_config["name"]),
        dataset_name=str(params["dataset"]["name"]),
        dataset_version=str(params["dataset"]["version"]),
        training_source=str(benchmark_config["dataset"]["training_source"]),
        synthetic_pack=str(benchmark_config["dataset"]["synthetic_pack"]),
        operating_recall_floor=training_config.operating_recall_floor,
        seed_results=report_seed_results,
    )
    return {
        "job_index_path": job_index_path,
        "report_artifacts": report_artifacts,
        "runs_root": runs_root,
        "dataset_materialized_dir": prepared["paths"]["dataset_materialized_dir"],
    }


def execute_benchmark_job(job: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Execute one anomaly benchmark job using the backend-specific command."""

    backend = str(job["model"]["backend"])
    run_dir = Path(str(job["run_dir"]))
    ensure_dir(run_dir)
    completed_result = load_completed_result(run_dir, expected_backend=backend)
    if completed_result is not None:
        return completed_result

    log_path = run_dir / "train.log"
    train_log_header = [
        f"started_at={datetime.now(timezone.utc).isoformat()}",
        f"model={job['model']['name']}",
        f"backend={backend}",
        f"fit_mode={job['model']['fit_mode']}",
        f"seed={job['seed']}",
    ]
    write_text(log_path, "\n".join(train_log_header) + "\n")
    command = build_backend_command(job, repo_root=repo_root)
    if command is None:
        result = AnomalySeedRunResult(
            model_name=str(job["model"]["name"]),
            display_name=str(job["model"]["display_name"]),
            seed=int(job["seed"]),
            status="not_implemented",
            backend=backend,
            fit_mode=str(job["model"]["fit_mode"]),
            model_artifact_path=(run_dir / "model.json").as_posix(),
            train_log_path=log_path.as_posix(),
            train_history_path=(run_dir / "train_history.csv").as_posix(),
            threshold_selection_path=(run_dir / "threshold_selection.json").as_posix(),
            val_scores_path=(run_dir / "val_scores.csv").as_posix(),
            test_scores_path=(run_dir / "test_scores.csv").as_posix(),
            val_metrics_path=(run_dir / "val_metrics.json").as_posix(),
            test_metrics_path=(run_dir / "test_metrics.json").as_posix(),
            generator_breakdown_path=(run_dir / "generator_breakdown.csv").as_posix(),
            anomaly_type_breakdown_path=(run_dir / "anomaly_type_breakdown.csv").as_posix(),
            severity_breakdown_path=(run_dir / "severity_breakdown.csv").as_posix(),
            notes=[f"backend `{backend}` is not implemented yet"],
        ).to_dict()
        write_json(run_dir / "result.json", result)
        return result

    returncode = stream_subprocess(command, log_path=log_path, cwd=repo_root)
    backend_result_path = run_dir / "result.json"
    if returncode == 0 and backend_result_path.exists():
        return read_json(backend_result_path, default={})
    result = AnomalySeedRunResult(
        model_name=str(job["model"]["name"]),
        display_name=str(job["model"]["display_name"]),
        seed=int(job["seed"]),
        status="failed",
        backend=backend,
        fit_mode=str(job["model"]["fit_mode"]),
        model_artifact_path=(run_dir / "model.json").as_posix(),
        train_log_path=log_path.as_posix(),
        train_history_path=(run_dir / "train_history.csv").as_posix(),
        threshold_selection_path=(run_dir / "threshold_selection.json").as_posix(),
        val_scores_path=(run_dir / "val_scores.csv").as_posix(),
        test_scores_path=(run_dir / "test_scores.csv").as_posix(),
        val_metrics_path=(run_dir / "val_metrics.json").as_posix(),
        test_metrics_path=(run_dir / "test_metrics.json").as_posix(),
        generator_breakdown_path=(run_dir / "generator_breakdown.csv").as_posix(),
        anomaly_type_breakdown_path=(run_dir / "anomaly_type_breakdown.csv").as_posix(),
        severity_breakdown_path=(run_dir / "severity_breakdown.csv").as_posix(),
        notes=[f"returncode={returncode}"],
    ).to_dict()
    write_json(run_dir / "result.json", result)
    return result


def build_backend_command(job: Mapping[str, Any], *, repo_root: Path) -> list[str] | None:
    """Build the backend command for one anomaly benchmark job."""

    backend = str(job["model"]["backend"])
    run_dir = Path(str(job["run_dir"]))
    command_prefix = [
        sys.executable,
        "-u",
        "-m",
    ]
    if backend == "placeholder":
        module_name = "towervision.anomaly.backends.placeholder_backend"
    elif backend == "anomalib":
        module_name = "towervision.anomaly.backends.anomalib_backend"
    elif backend == "repo_cutpaste":
        module_name = "towervision.anomaly.backends.cutpaste_backend"
    else:
        return None
    return [
        *command_prefix,
        module_name,
        "--job",
        (run_dir / "job.json").as_posix(),
    ]


def stream_subprocess(command: Sequence[str], *, log_path: Path, cwd: Path) -> int:
    """Run a subprocess while teeing stdout and stderr to terminal and file."""

    env = os.environ.copy()
    src_path = (cwd / "src").as_posix()
    env["PYTHONPATH"] = src_path if "PYTHONPATH" not in env else f"{src_path}:{env['PYTHONPATH']}"
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


def _resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path
