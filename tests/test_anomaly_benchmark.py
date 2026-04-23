from __future__ import annotations

from pathlib import Path

from PIL import Image

from towervision.anomaly.benchmark import (
    build_backend_command,
    collect_benchmark_seed_results,
    load_completed_result,
)
from towervision.anomaly.benchmark_dataset import materialize_anomaly_benchmark_dataset, read_dataset_manifest
from towervision.anomaly.benchmark_reporting import aggregate_seed_results
from towervision.utils.io import write_json


def test_materialize_anomaly_benchmark_dataset_builds_train_val_test_views(tmp_path: Path) -> None:
    images_manifest = tmp_path / "images.json"
    annotations_manifest = tmp_path / "annotations.json"
    splits_path = tmp_path / "splits.json"
    records_path = tmp_path / "synthetic" / "records.csv"
    source_candidates_path = records_path.parent / "source_candidates.csv"

    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    image_a = source_dir / "a.jpg"
    image_b = source_dir / "b.jpg"
    Image.new("RGB", (64, 64), color="white").save(image_a)
    Image.new("RGB", (64, 64), color="gray").save(image_b)

    val_source_crop = records_path.parent / "source_crops" / "val" / "b__2__isoladores.png"
    val_source_crop.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color="white").save(val_source_crop)

    anomaly_output = records_path.parent / "generated" / "chatgpt" / "val_1_b__2_gpt.png"
    anomaly_output.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color="black").save(anomaly_output)
    anomaly_mask = records_path.parent / "masks" / "val" / "val_1_b__2__chatgpt__mask.png"
    anomaly_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=255).save(anomaly_mask)

    write_json(
        images_manifest,
        [
            {"id": "a", "path": image_a.as_posix(), "width": 64, "height": 64, "metadata": {}},
            {"id": "b", "path": image_b.as_posix(), "width": 64, "height": 64, "metadata": {}},
        ],
    )
    write_json(
        annotations_manifest,
        [
            {"id": "1", "image_id": "a", "bbox": [10, 10, 20, 20], "label": "isoladores", "source": "gt"},
            {"id": "2", "image_id": "b", "bbox": [12, 12, 20, 20], "label": "isoladores", "source": "gt"},
        ],
    )
    write_json(splits_path, {"train": ["a"], "val": ["b"], "test": []})
    source_candidates_path.write_text(
        "source_crop_id,source_split,source_image_id,source_image_path,annotation_id,label,bbox_x,bbox_y,bbox_width,bbox_height,bbox_area,crop_width,crop_height,padding,source_crop_path\n"
        f"b__2,val,b,{image_b.as_posix()},2,isoladores,12,12,20,20,400,32,32,64,{val_source_crop.as_posix()}\n",
        encoding="utf-8",
    )
    records_path.write_text(
        "record_id,pair_id,source_image_id,source_image_path,source_crop_path,source_split,generator_family,generator_model,anomaly_scope,anomaly_type,severity,output_image_path,mask_path,prompt_path,accepted_for_benchmark,notes\n"
        f"val_1_b__2__chatgpt,val_1_b__2,b,{image_b.as_posix()},{val_source_crop.as_posix()},val,chatgpt,,isoladores,crack,moderate,{anomaly_output.as_posix()},{anomaly_mask.as_posix()},,true,\n",
        encoding="utf-8",
    )

    artifacts = materialize_anomaly_benchmark_dataset(
        images_manifest_path=images_manifest,
        annotations_manifest_path=annotations_manifest,
        splits_path=splits_path,
        synthetic_records_path=records_path,
        output_dir=tmp_path / "benchmark_dataset",
    )

    train_rows = read_dataset_manifest(artifacts.split_to_manifest_path["train"])
    val_rows = read_dataset_manifest(artifacts.split_to_manifest_path["val"])
    assert artifacts.crop_padding == 64
    assert len(train_rows) == 1
    assert len(val_rows) == 2
    assert sum(row.label for row in val_rows) == 1


def test_aggregate_anomaly_seed_results() -> None:
    aggregated = aggregate_seed_results(
        [
            {
                "model_name": "patchcore",
                "display_name": "PatchCore",
                "status": "completed",
                "val_metrics": {"roi_auroc": 0.80, "roi_auprc": 0.70, "f1": 0.60, "precision": 0.75, "recall": 0.50, "accuracy": 0.90, "threshold": 0.42},
                "test_metrics": {"roi_auroc": 0.78, "roi_auprc": 0.68, "f1": 0.58, "precision": 0.70, "recall": 0.50, "accuracy": 0.88, "threshold": 0.42},
            },
            {
                "model_name": "patchcore",
                "display_name": "PatchCore",
                "status": "completed",
                "val_metrics": {"roi_auroc": 0.82, "roi_auprc": 0.72, "f1": 0.62, "precision": 0.78, "recall": 0.52, "accuracy": 0.91, "threshold": 0.44},
                "test_metrics": {"roi_auroc": 0.79, "roi_auprc": 0.69, "f1": 0.59, "precision": 0.71, "recall": 0.51, "accuracy": 0.89, "threshold": 0.44},
            },
        ]
    )

    assert len(aggregated) == 1
    assert aggregated[0]["model_name"] == "patchcore"
    assert aggregated[0]["test_roi_auroc_mean"] == 0.785


def test_collect_anomaly_benchmark_seed_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "patchcore" / "seed_42"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "result.json", {"model_name": "patchcore", "seed": 42, "status": "completed"})

    collected = collect_benchmark_seed_results(tmp_path)
    loaded = load_completed_result(run_dir)

    assert len(collected) == 1
    assert loaded is not None
    assert loaded["status"] == "completed"


def test_build_backend_command_routes_real_anomaly_backends(tmp_path: Path) -> None:
    run_dir = tmp_path / "cutpaste" / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "job.json").write_text("{}", encoding="utf-8")

    def make_job(backend: str) -> dict[str, object]:
        return {
            "model": {"backend": backend},
            "run_dir": run_dir.as_posix(),
        }

    patchcore_command = build_backend_command(make_job("anomalib"), repo_root=tmp_path)
    cutpaste_command = build_backend_command(make_job("repo_cutpaste"), repo_root=tmp_path)
    placeholder_command = build_backend_command(make_job("placeholder"), repo_root=tmp_path)

    assert patchcore_command is not None
    assert patchcore_command[3] == "towervision.anomaly.backends.anomalib_backend"
    assert cutpaste_command is not None
    assert cutpaste_command[3] == "towervision.anomaly.backends.cutpaste_backend"
    assert placeholder_command is not None
    assert placeholder_command[3] == "towervision.anomaly.backends.placeholder_backend"
