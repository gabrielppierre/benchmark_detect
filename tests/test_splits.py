from pathlib import Path

from towervision.data.load import ImageRecord
from towervision.data.splits import (
    build_temporal_groups,
    generate_official_grouped_split,
    generate_splits,
    make_time_bucket_group_id,
)


def test_generate_splits_has_no_leakage() -> None:
    image_ids = [f"img_{index:02d}" for index in range(12)]

    splits = generate_splits(image_ids, train_ratio=0.5, val_ratio=0.25, seed=7)

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(image_ids)


def test_make_time_bucket_group_id_uses_30_second_bucket() -> None:
    image_id = "DJI_20250911105044_0101_V"

    group_id = make_time_bucket_group_id(image_id, bucket_seconds=30)

    assert group_id == "20250911_1050_30_30s"


def test_generate_official_grouped_split_keeps_group_members_together() -> None:
    image_ids = [
        "DJI_20250911105000_0001_V",
        "DJI_20250911105005_0002_V",
        "DJI_20250911105030_0003_V",
        "DJI_20250911105035_0004_V",
        "DJI_20250911105100_0005_V",
        "DJI_20250911105105_0006_V",
    ]
    images = [
        ImageRecord(id=image_id, path=Path(f"/tmp/{image_id}.jpg"))
        for image_id in image_ids
    ]

    splits, metadata = generate_official_grouped_split(
        images,
        train_ratio=0.34,
        val_ratio=0.33,
        bucket_seconds=30,
    )
    _, image_to_group = build_temporal_groups(image_ids, bucket_seconds=30)
    image_to_split = {
        image_id: split_name
        for split_name, split_image_ids in splits.items()
        for image_id in split_image_ids
    }

    for image_id, group_id in image_to_group.items():
        grouped_images = [
            candidate_id
            for candidate_id, candidate_group_id in image_to_group.items()
            if candidate_group_id == group_id
        ]
        grouped_splits = {image_to_split[candidate_id] for candidate_id in grouped_images}
        assert len(grouped_splits) == 1

    assert metadata["group_strategy"] == "filename_time_bucket"
