"""Unified orchestration for the fair detection benchmark."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from towervision.detectors.benchmark_dataset import materialize_detection_benchmark_dataset
from towervision.detectors.benchmark_reporting import persist_benchmark_report
from towervision.detectors.benchmark_types import (
    BenchmarkDatasetArtifacts,
    BenchmarkModelConfig,
    BenchmarkTrainingConfig,
    SeedRunResult,
)
from towervision.utils.io import ensure_dir, read_json, read_yaml, write_json


def load_benchmark_experiment_config(path: Path) -> dict[str, Any]:
    """Load the fair benchmark experiment YAML."""

    payload = read_yaml(path, default={})
    if not isinstance(payload, dict):
        raise ValueError(f"invalid benchmark config: {path}")
    return payload


def parse_training_config(raw: Mapping[str, Any]) -> BenchmarkTrainingConfig:
    """Build the shared training config from YAML."""

    seeds = list(raw.get("seeds", []))
    num_seeds = int(raw.get("num_seeds", len(seeds)))
    if not seeds:
        seeds = [42 + (index * 10) for index in range(num_seeds)]
    return BenchmarkTrainingConfig(
        img_size=int(raw["img_size"]),
        max_epochs=int(raw["max_epochs"]),
        validate_every=int(raw["validate_every"]),
        save_best=bool(raw["save_best"]),
        save_last=bool(raw["save_last"]),
        early_stopping=bool(raw["early_stopping"]),
        monitor=str(raw["monitor"]),
        mode=str(raw["mode"]),
        patience=int(raw["patience"]),
        min_epochs=int(raw["min_epochs"]),
        num_seeds=num_seeds,
        seeds=[int(seed) for seed in seeds[:num_seeds]],
        recall_floor_isoladores=float(raw["recall_floor_isoladores"]),
        augmentations=dict(raw.get("augmentations", {})),
    )


def parse_model_configs(raw_models: Sequence[Mapping[str, Any]]) -> list[BenchmarkModelConfig]:
    """Build per-model configs from YAML."""

    return [
        BenchmarkModelConfig(
            name=str(raw_model["name"]),
            display_name=str(raw_model.get("display_name", raw_model["name"])),
            backend=str(raw_model["backend"]),
            family=str(raw_model["family"]),
            pretrained_weights=str(raw_model["pretrained_weights"]),
            batch_size=int(raw_model["batch_size"]),
            num_workers=int(raw_model.get("num_workers", 4)),
            confidence_threshold=float(raw_model.get("confidence_threshold", 0.001)),
            nms_iou_threshold=float(raw_model.get("nms_iou_threshold", 0.6)),
            notes=str(raw_model.get("notes", "")),
            extra=dict(raw_model.get("extra", {})),
        )
        for raw_model in raw_models
    ]


def resolve_benchmark_paths(
    *,
    repo_root: Path,
    params: Mapping[str, Any],
    benchmark_name: str,
) -> dict[str, Path]:
    """Resolve the benchmark output paths from the active dataset version."""

    dataset_name = str(params["dataset"]["name"])
    dataset_version = str(params["dataset"]["version"])
    return {
        "dataset_materialized_dir": (
            repo_root
            / "data"
            / "interim"
            / dataset_name
            / dataset_version
            / "detection_benchmark"
            / benchmark_name
            / "dataset"
        ),
        "runs_root": (
            repo_root
            / "runs"
            / "detectors"
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
            / "detection_benchmark"
            / benchmark_name
        ),
    }


def prepare_detection_benchmark(
    *,
    repo_root: Path,
    params: Mapping[str, Any],
    benchmark_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare the materialized dataset and output directories for the benchmark."""

    benchmark_name = str(benchmark_config["name"])
    paths = resolve_benchmark_paths(repo_root=repo_root, params=params, benchmark_name=benchmark_name)
    class_names = list(benchmark_config["classes"])

    artifacts = materialize_detection_benchmark_dataset(
        images_manifest_path=_resolve_repo_path(repo_root, params["paths"]["cleaned_images_manifest"]),
        annotations_manifest_path=_resolve_repo_path(repo_root, params["paths"]["cleaned_annotations_manifest"]),
        splits_path=_resolve_repo_path(repo_root, params["paths"]["splits_path"]),
        output_dir=paths["dataset_materialized_dir"],
        class_names=class_names,
    )

    ensure_dir(paths["runs_root"])
    ensure_dir(paths["reports_root"])

    return {
        "paths": paths,
        "dataset_artifacts": artifacts,
        "class_names": class_names,
    }


def build_job_specs(
    *,
    repo_root: Path,
    params: Mapping[str, Any],
    benchmark_config: Mapping[str, Any],
    dataset_artifacts: BenchmarkDatasetArtifacts,
    model_configs: Sequence[BenchmarkModelConfig],
    training_config: BenchmarkTrainingConfig,
    runs_root: Path,
    selected_models: Sequence[str] | None = None,
    selected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Create reproducible job specs for each model and seed."""

    dataset_name = str(params["dataset"]["name"])
    dataset_version = str(params["dataset"]["version"])
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
                    "dataset_name": dataset_name,
                    "dataset_version": dataset_version,
                    "split_name": benchmark_config["split_name"],
                    "critical_class": benchmark_config["critical_class"],
                    "class_names": list(benchmark_config["classes"]),
                    "dataset_views": {
                        "materialized_root": dataset_artifacts.root_dir.as_posix(),
                        "coco_root": dataset_artifacts.coco_root.as_posix(),
                        "coco_annotations": {
                            split_name: path.as_posix()
                            for split_name, path in dataset_artifacts.split_to_annotation_path.items()
                        },
                        "coco_images": {
                            split_name: path.as_posix()
                            for split_name, path in dataset_artifacts.split_to_image_dir.items()
                        },
                        "ultralytics_root": dataset_artifacts.ultralytics_root.as_posix(),
                        "ultralytics_dataset_yaml": dataset_artifacts.ultralytics_dataset_yaml.as_posix(),
                    },
                    "model": {
                        "name": model.name,
                        "display_name": model.display_name,
                        "backend": model.backend,
                        "family": model.family,
                        "pretrained_weights": model.pretrained_weights,
                        "batch_size": model.batch_size,
                        "num_workers": model.num_workers,
                        "confidence_threshold": model.confidence_threshold,
                        "nms_iou_threshold": model.nms_iou_threshold,
                        "notes": model.notes,
                        "extra": model.extra,
                    },
                    "training": {
                        "img_size": training_config.img_size,
                        "max_epochs": training_config.max_epochs,
                        "validate_every": training_config.validate_every,
                        "save_best": training_config.save_best,
                        "save_last": training_config.save_last,
                        "early_stopping": training_config.early_stopping,
                        "monitor": training_config.monitor,
                        "mode": training_config.mode,
                        "patience": training_config.patience,
                        "min_epochs": training_config.min_epochs,
                        "seed": seed,
                        "augmentations": training_config.augmentations,
                    },
                    "run_dir": run_dir.as_posix(),
                    "seed": seed,
                }
            )
    return jobs


def write_job_specs(job_specs: Sequence[Mapping[str, Any]], *, runs_root: Path) -> Path:
    """Persist the job index and one job spec per run directory."""

    for job in job_specs:
        run_dir = Path(str(job["run_dir"]))
        ensure_dir(run_dir)
        write_json(run_dir / "job.json", dict(job))
    index_path = runs_root / "job_index.json"
    write_json(index_path, list(job_specs))
    return index_path


def collect_benchmark_seed_results(runs_root: Path) -> list[dict[str, Any]]:
    """Collect every structured result already persisted under the benchmark runs root."""

    results_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for result_path in sorted(runs_root.glob("*/seed_*/result.json")):
        result = read_json(result_path, default={})
        if not isinstance(result, dict):
            continue
        model_name = str(result.get("model_name", result_path.parent.parent.name))
        seed = int(result.get("seed", result_path.parent.name.removeprefix("seed_")))
        results_by_key[(model_name, seed)] = result
    return list(results_by_key.values())


def load_completed_result(run_dir: Path) -> dict[str, Any] | None:
    """Return an existing completed result for one run directory when available."""

    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None
    result = read_json(result_path, default={})
    if not isinstance(result, dict):
        return None
    if str(result.get("status")) != "completed":
        return None
    return result


def run_detection_benchmark(
    *,
    repo_root: Path,
    params_path: Path,
    benchmark_config_path: Path,
    execute: bool = False,
    selected_models: Sequence[str] | None = None,
    selected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Prepare the benchmark and optionally execute the queued jobs."""

    params = read_yaml(params_path)
    benchmark_config = load_benchmark_experiment_config(benchmark_config_path)
    training_config = parse_training_config(benchmark_config["training"])
    model_configs = parse_model_configs(benchmark_config["models"])

    prepared = prepare_detection_benchmark(
        repo_root=repo_root,
        params=params,
        benchmark_config=benchmark_config,
    )
    runs_root = prepared["paths"]["runs_root"]
    job_specs = build_job_specs(
        repo_root=repo_root,
        params=params,
        benchmark_config=benchmark_config,
        dataset_artifacts=prepared["dataset_artifacts"],
        model_configs=model_configs,
        training_config=training_config,
        runs_root=runs_root,
        selected_models=selected_models,
        selected_seeds=selected_seeds,
    )
    job_index_path = write_job_specs(job_specs, runs_root=runs_root)

    seed_results: list[dict[str, Any]] = []
    if execute:
        for job in job_specs:
            seed_results.append(execute_benchmark_job(job, repo_root=repo_root))
    else:
        for job in job_specs:
            run_dir = Path(str(job["run_dir"]))
            seed_results.append(
                SeedRunResult(
                    model_name=str(job["model"]["name"]),
                    display_name=str(job["model"]["display_name"]),
                    seed=int(job["seed"]),
                    status="planned",
                    best_epoch=None,
                    best_checkpoint_path=None,
                    train_log_path=(run_dir / "train.log").as_posix(),
                    epoch_metrics_path=(run_dir / "epoch_metrics.csv").as_posix(),
                    notes=["job planned but not executed"],
                ).to_dict()
            )

    report_seed_results = (
        collect_benchmark_seed_results(runs_root) if execute else seed_results
    )

    report_artifacts = persist_benchmark_report(
        root_dir=prepared["paths"]["reports_root"],
        benchmark_name=str(benchmark_config["name"]),
        dataset_name=str(params["dataset"]["name"]),
        dataset_version=str(params["dataset"]["version"]),
        split_name=str(benchmark_config["split_name"]),
        classes=prepared["class_names"],
        critical_class=str(benchmark_config["critical_class"]),
        recall_floor_isoladores=training_config.recall_floor_isoladores,
        seed_results=report_seed_results,
    )

    return {
        "job_index_path": job_index_path,
        "report_artifacts": report_artifacts,
        "runs_root": runs_root,
        "dataset_materialized_dir": prepared["paths"]["dataset_materialized_dir"],
    }


def execute_benchmark_job(job: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Execute one benchmark job using the backend-specific command."""

    backend = str(job["model"]["backend"])
    run_dir = Path(str(job["run_dir"]))
    ensure_dir(run_dir)
    completed_result = load_completed_result(run_dir)
    if completed_result is not None:
        return completed_result
    log_path = run_dir / "train.log"
    train_log_header = [
        f"started_at={datetime.now(timezone.utc).isoformat()}",
        f"model={job['model']['name']}",
        f"backend={backend}",
        f"seed={job['seed']}",
    ]
    write_text(log_path, "\n".join(train_log_header) + "\n")

    command = build_backend_command(job, repo_root=repo_root)
    if command is None:
        result = SeedRunResult(
            model_name=str(job["model"]["name"]),
            display_name=str(job["model"]["display_name"]),
            seed=int(job["seed"]),
            status="not_implemented",
            best_epoch=None,
            best_checkpoint_path=None,
            train_log_path=log_path.as_posix(),
            epoch_metrics_path=(run_dir / "epoch_metrics.csv").as_posix(),
            notes=[f"backend `{backend}` is not implemented yet"],
        ).to_dict()
        write_json(run_dir / "result.json", result)
        return result

    returncode = stream_subprocess(command, log_path=log_path, cwd=repo_root)
    backend_result_path = run_dir / "result.json"
    if returncode == 0 and backend_result_path.exists():
        return read_json(backend_result_path, default={})
    result = SeedRunResult(
        model_name=str(job["model"]["name"]),
        display_name=str(job["model"]["display_name"]),
        seed=int(job["seed"]),
        status="completed" if returncode == 0 else "failed",
        best_epoch=None,
        best_checkpoint_path=None,
        train_log_path=log_path.as_posix(),
        epoch_metrics_path=(run_dir / "epoch_metrics.csv").as_posix(),
        notes=[
            f"returncode={returncode}",
            "backend execution completed but structured metric ingestion still depends on backend adapters",
        ],
    ).to_dict()
    write_json(run_dir / "result.json", result)
    return result


def build_backend_command(job: Mapping[str, Any], *, repo_root: Path) -> list[str] | None:
    """Build a backend command for one model/seed job."""

    backend = str(job["model"]["backend"])
    run_dir = Path(str(job["run_dir"]))
    if backend == "placeholder":
        return None
    if backend == "ultralytics":
        return [
            sys.executable,
            "-u",
            "-m",
            "towervision.detectors.backends.ultralytics_backend",
            "--job",
            (run_dir / "job.json").as_posix(),
        ]
    if backend == "torchvision":
        return [
            sys.executable,
            "-u",
            "-m",
            "towervision.detectors.backends.torchvision_backend",
            "--job",
            (run_dir / "job.json").as_posix(),
        ]
    if backend == "yolox":
        return [
            sys.executable,
            "-u",
            "-m",
            "towervision.detectors.backends.yolox_backend",
            "--job",
            (run_dir / "job.json").as_posix(),
        ]
    if backend == "transformers":
        return [
            sys.executable,
            "-u",
            "-m",
            "towervision.detectors.backends.transformers_backend",
            "--job",
            (run_dir / "job.json").as_posix(),
        ]
    return None


def stream_subprocess(command: Sequence[str], *, log_path: Path, cwd: Path) -> int:
    """Run a subprocess while teeing stdout/stderr to terminal and file."""

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
