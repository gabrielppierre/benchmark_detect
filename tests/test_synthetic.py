from pathlib import Path

from PIL import Image

from towervision.data.load import AnnotationRecord, ImageRecord
from towervision.data.synthetic import (
    accept_synthetic_records_for_benchmark,
    build_synthetic_pack_paths,
    export_synthetic_source_crops,
    import_roboflow_segmentation_masks,
    initialize_controlled_synthetic_pack,
    materialize_shortlist_bundle,
    read_csv_rows,
    render_synthetic_overlay_contact_sheet,
    sync_records_from_generated_outputs,
    write_source_csv,
    RECORD_FIELDS,
)


def test_initialize_controlled_synthetic_pack_creates_expected_layout(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "tower_vision" / "v2026-04-16" / "extracted" / "imagens_torres_300"
    raw_root.mkdir(parents=True)

    paths = initialize_controlled_synthetic_pack(
        tmp_path,
        dataset_name="tower_vision",
        dataset_version="v2026-04-16",
        raw_dataset_root=raw_root,
    )

    assert paths.manifest_path.exists()
    assert paths.records_path.exists()
    assert paths.readme_path.exists()
    assert paths.generated_dirs["chatgpt"].exists()
    assert paths.generated_dirs["gemini"].exists()
    assert paths.prompt_dirs["chatgpt"].exists()
    assert paths.prompt_dirs["gemini"].exists()
    assert paths.source_crop_dirs["val"].exists()
    assert paths.source_crop_dirs["test"].exists()
    assert paths.source_candidates_path.exists()
    assert paths.source_shortlist_path.exists()
    assert "record_id" in paths.records_path.read_text(encoding="utf-8")


def test_export_synthetic_source_crops_creates_candidates_and_shortlist(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "tower_vision" / "v2026-04-16" / "extracted" / "imagens_torres_300"
    image_path = raw_root / "images" / "default" / "DJI_20250911105200_0209_V.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), color="white").save(image_path)

    paths = initialize_controlled_synthetic_pack(
        tmp_path,
        dataset_name="tower_vision",
        dataset_version="v2026-04-16",
        raw_dataset_root=raw_root,
    )
    images_by_id = {
        "DJI_20250911105200_0209_V": ImageRecord(
            id="DJI_20250911105200_0209_V",
            path=image_path,
            width=512,
            height=512,
        )
    }
    annotations = [
        AnnotationRecord(
            id="ann-1",
            image_id="DJI_20250911105200_0209_V",
            bbox=(100.0, 100.0, 40.0, 200.0),
            label="isoladores",
        )
    ]
    splits = {"val": ["DJI_20250911105200_0209_V"], "test": []}

    candidates, shortlist = export_synthetic_source_crops(
        paths,
        images_by_id,
        annotations,
        splits,
        padding=32,
        shortlist_per_split=5,
    )

    assert len(candidates) == 1
    assert len(shortlist) == 1
    assert Path(candidates[0]["source_crop_path"]).exists()
    assert "source_crop_id" in paths.source_candidates_path.read_text(encoding="utf-8")
    assert "shortlist_rank" in paths.source_shortlist_path.read_text(encoding="utf-8")

    copied_rows = materialize_shortlist_bundle(paths)

    assert len(copied_rows) == 1
    assert Path(copied_rows[0]["bundle_path"]).exists()

    chatgpt_output = paths.generated_dirs["chatgpt"] / "val_1_DJI_20250911105200_0209_V__ann-1_gpt.png"
    gemini_output = paths.generated_dirs["gemini"] / "val_1_DJI_20250911105200_0209_V__ann-1_gemini.png"
    Image.new("RGB", (128, 128), color="gray").save(chatgpt_output)
    Image.new("RGB", (128, 128), color="gray").save(gemini_output)

    records = sync_records_from_generated_outputs(build_synthetic_pack_paths(
        tmp_path,
        dataset_name="tower_vision",
        dataset_version="v2026-04-16",
        pack_name="anomaly_controlled_v1",
    ))

    assert len(records) == 2
    assert records[0]["anomaly_type"] == "crack"
    assert records[0]["severity"] == "moderate"
    assert records[0]["prompt_path"].endswith("01_crack.md")
    assert records[0]["generator_family"] == "chatgpt"
    assert records[0]["accepted_for_benchmark"] == "false"
    assert records[0]["notes"] == "pending_mask_annotation"


def test_import_roboflow_masks_uses_repository_split_not_export_train(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "tower_vision" / "v2026-04-16" / "extracted" / "imagens_torres_300"
    image_path = raw_root / "images" / "default" / "DJI_20250911105200_0209_V.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), color="white").save(image_path)

    paths = initialize_controlled_synthetic_pack(
        tmp_path,
        dataset_name="tower_vision",
        dataset_version="v2026-04-16",
        raw_dataset_root=raw_root,
    )
    images_by_id = {
        "DJI_20250911105200_0209_V": ImageRecord(
            id="DJI_20250911105200_0209_V",
            path=image_path,
            width=512,
            height=512,
        )
    }
    annotations = [
        AnnotationRecord(
            id="ann-1",
            image_id="DJI_20250911105200_0209_V",
            bbox=(100.0, 100.0, 40.0, 200.0),
            label="isoladores",
        )
    ]
    splits = {"val": ["DJI_20250911105200_0209_V"], "test": []}
    export_synthetic_source_crops(
        paths,
        images_by_id,
        annotations,
        splits,
        padding=32,
        shortlist_per_split=1,
    )
    materialize_shortlist_bundle(paths)

    generated_output = paths.generated_dirs["chatgpt"] / "val_1_DJI_20250911105200_0209_V__ann-1_gpt.png"
    Image.new("RGB", (8, 8), color="gray").save(generated_output)
    sync_records_from_generated_outputs(
        build_synthetic_pack_paths(
            tmp_path,
            dataset_name="tower_vision",
            dataset_version="v2026-04-16",
            pack_name="anomaly_controlled_v1",
        )
    )

    export_root = tmp_path / "dataset_segmentation_roboflow"
    train_dir = export_root / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "_annotations.coco.json").write_text(
        """{
  "images": [
    {
      "id": 1,
      "file_name": "val_1_DJI_20250911105200_0209_V__ann-1_gpt_png.rf.abc123.jpg",
      "width": 8,
      "height": 8
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [2, 2, 4, 4],
      "area": 16,
      "segmentation": [[2, 2, 6, 2, 6, 6, 2, 6]],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "anomaly"}
  ]
}
""",
        encoding="utf-8",
    )

    summary = import_roboflow_segmentation_masks(paths, export_root)
    imported_rows = read_csv_rows(paths.records_path)

    assert summary["imported_mask_count"] == 1
    assert summary["matched_record_count"] == 1
    assert imported_rows[0]["mask_path"].endswith("__mask.png")
    assert "/masks/val/" in imported_rows[0]["mask_path"]
    assert "mask_imported_from_roboflow=true" in imported_rows[0]["notes"]
    assert "pending_mask_annotation" not in imported_rows[0]["notes"]
    assert Path(imported_rows[0]["mask_path"]).exists()


def test_accept_synthetic_records_marks_only_valid_rows(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "tower_vision" / "v2026-04-16" / "extracted" / "imagens_torres_300"
    paths = initialize_controlled_synthetic_pack(
        tmp_path,
        dataset_name="tower_vision",
        dataset_version="v2026-04-16",
        raw_dataset_root=raw_root,
    )
    valid_mask = paths.masks_root / "val" / "valid_mask.png"
    valid_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=255).save(valid_mask)

    write_source_csv(
        paths.records_path,
        RECORD_FIELDS,
        [
            {
                "record_id": "valid_row",
                "accepted_for_benchmark": "false",
                "mask_path": valid_mask.as_posix(),
                "notes": "review_valid=true",
            },
            {
                "record_id": "invalid_row",
                "accepted_for_benchmark": "false",
                "mask_path": "",
                "notes": "review_valid=false",
            },
        ],
    )

    summary = accept_synthetic_records_for_benchmark(paths)
    rows = read_csv_rows(paths.records_path)

    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1
    assert rows[0]["accepted_for_benchmark"] == "true"
    assert "accepted_for_benchmark_by_curated_review=true" in rows[0]["notes"]
    assert rows[1]["accepted_for_benchmark"] == "false"


def test_render_synthetic_overlay_contact_sheet_creates_png(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "tower_vision" / "v2026-04-16" / "extracted" / "imagens_torres_300"
    paths = initialize_controlled_synthetic_pack(
        tmp_path,
        dataset_name="tower_vision",
        dataset_version="v2026-04-16",
        raw_dataset_root=raw_root,
    )
    overlay_root = tmp_path / "reports" / "figures"
    overlay_path = overlay_root / "val" / "chatgpt" / "row_1__overlay.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 60), color="white").save(overlay_path)

    write_source_csv(
        paths.records_path,
        RECORD_FIELDS,
        [
            {
                "record_id": "row_1",
                "source_split": "val",
                "generator_family": "chatgpt",
                "severity": "moderate",
                "notes": "review_suggested_severity=moderate",
            }
        ],
    )

    summary = render_synthetic_overlay_contact_sheet(
        paths,
        overlay_root=overlay_root,
        output_path=overlay_root / "contact_sheet.png",
        columns=2,
    )

    assert summary["overlay_item_count"] == 1
    assert Path(summary["contact_sheet_path"]).exists()
