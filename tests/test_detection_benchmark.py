from __future__ import annotations

from pathlib import Path

from towervision.detectors.benchmark import collect_benchmark_seed_results, load_completed_result
from towervision.detectors.benchmark_dataset import materialize_detection_benchmark_dataset
from towervision.detectors.benchmark_reporting import (
    aggregate_seed_results,
    select_detector_for_anomaly,
)
from towervision.utils.io import write_json


def test_materialize_detection_benchmark_dataset_exports_split_views(tmp_path: Path) -> None:
    images_manifest = tmp_path / "images.json"
    annotations_manifest = tmp_path / "annotations.json"
    splits_path = tmp_path / "splits.json"

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    image_a = source_dir / "a.jpg"
    image_b = source_dir / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    write_json(
        images_manifest,
        [
            {
                "id": "a",
                "path": image_a.as_posix(),
                "width": 100,
                "height": 100,
                "metadata": {},
            },
            {
                "id": "b",
                "path": image_b.as_posix(),
                "width": 200,
                "height": 100,
                "metadata": {},
            },
        ],
    )
    write_json(
        annotations_manifest,
        [
            {
                "id": "1",
                "image_id": "a",
                "bbox": [10, 10, 50, 50],
                "label": "torre",
                "source": "gt",
                "metadata": {"area": 2500},
            },
            {
                "id": "2",
                "image_id": "b",
                "bbox": [20, 10, 40, 20],
                "label": "isoladores",
                "source": "gt",
                "metadata": {"area": 800},
            },
        ],
    )
    write_json(splits_path, {"train": ["a"], "val": ["b"], "test": []})

    artifacts = materialize_detection_benchmark_dataset(
        images_manifest_path=images_manifest,
        annotations_manifest_path=annotations_manifest,
        splits_path=splits_path,
        output_dir=tmp_path / "benchmark_dataset",
        class_names=("torre", "isoladores"),
    )

    assert artifacts.split_to_annotation_path["train"].exists()
    assert artifacts.split_to_annotation_path["val"].exists()
    assert artifacts.ultralytics_dataset_yaml.exists()
    assert (artifacts.split_to_yolo_label_dir["train"] / "a.txt").exists()
    assert (artifacts.split_to_yolo_label_dir["val"] / "b.txt").exists()


def test_aggregate_seed_results_and_selection() -> None:
    seed_results = [
        {
            "model_name": "yolo11s",
            "display_name": "YOLO11s",
            "status": "completed",
            "val_best_metrics": {
                "val_map50_95": 0.42,
                "AP50_95_isoladores": 0.36,
                "Recall_isoladores": 0.87,
            },
            "test_metrics": {
                "mAP50_95": 0.40,
                "AP50_95_isoladores": 0.34,
                "Recall_isoladores": 0.84,
            },
        },
        {
            "model_name": "fasterrcnn_resnet50_fpn_v2",
            "display_name": "Faster R-CNN ResNet50-FPN v2",
            "status": "completed",
            "val_best_metrics": {
                "val_map50_95": 0.39,
                "AP50_95_isoladores": 0.35,
                "Recall_isoladores": 0.90,
            },
            "test_metrics": {
                "mAP50_95": 0.41,
                "AP50_95_isoladores": 0.33,
                "Recall_isoladores": 0.82,
            },
        },
    ]

    aggregated = aggregate_seed_results(seed_results)
    selected = select_detector_for_anomaly(aggregated, recall_floor_isoladores=0.85)

    assert len(aggregated) == 2
    assert selected is not None
    assert selected["model_name"] == "yolo11s"


def test_collect_benchmark_seed_results_merges_existing_results(tmp_path: Path) -> None:
    first_result_dir = tmp_path / "yolo11s" / "seed_42"
    second_result_dir = tmp_path / "yolo11s" / "seed_52"
    first_result_dir.mkdir(parents=True)
    second_result_dir.mkdir(parents=True)

    write_json(first_result_dir / "result.json", {"model_name": "yolo11s", "seed": 42, "status": "completed"})
    write_json(second_result_dir / "result.json", {"model_name": "yolo11s", "seed": 52, "status": "completed"})

    collected = collect_benchmark_seed_results(tmp_path)

    assert len(collected) == 2
    assert {(item["model_name"], item["seed"]) for item in collected} == {
        ("yolo11s", 42),
        ("yolo11s", 52),
    }


def test_load_completed_result_only_returns_completed_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "yolo11s" / "seed_42"
    run_dir.mkdir(parents=True)
    result_path = run_dir / "result.json"

    write_json(result_path, {"model_name": "yolo11s", "seed": 42, "status": "failed"})
    assert load_completed_result(run_dir) is None

    write_json(result_path, {"model_name": "yolo11s", "seed": 42, "status": "completed"})
    loaded = load_completed_result(run_dir)

    assert loaded is not None
    assert loaded["status"] == "completed"
