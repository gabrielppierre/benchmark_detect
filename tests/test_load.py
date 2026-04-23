from pathlib import Path

from towervision.data.load import load_coco_dataset
from towervision.utils.io import write_json


def test_load_coco_dataset_normalizes_image_and_annotation_ids(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    annotations_path = tmp_path / "instances.json"

    write_json(
        annotations_path,
        {
            "images": [
                {
                    "id": 10,
                    "file_name": "sample/image_001.jpg",
                    "width": 100,
                    "height": 80,
                }
            ],
            "annotations": [
                {
                    "id": 7,
                    "image_id": 10,
                    "category_id": 2,
                    "bbox": [1.5, 2.5, 20.0, 10.0],
                    "area": 200.0,
                    "iscrowd": 0,
                }
            ],
            "categories": [
                {"id": 2, "name": "isoladores"},
            ],
        },
    )

    images, annotations = load_coco_dataset(images_dir, annotations_path)

    assert len(images) == 1
    assert images[0].id == "sample/image_001"
    assert images[0].path == images_dir / "sample/image_001.jpg"
    assert len(annotations) == 1
    assert annotations[0].image_id == "sample/image_001"
    assert annotations[0].label == "isoladores"
    assert annotations[0].bbox == (1.5, 2.5, 20.0, 10.0)
