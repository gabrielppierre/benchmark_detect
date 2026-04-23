from PIL import Image

from towervision.pipelines.crop_from_gt import crop_bbox


def test_crop_bbox_returns_expected_size() -> None:
    image = Image.new("RGB", (20, 20), color="white")

    crop = crop_bbox(image, (5, 6, 7, 8))

    assert crop.size == (7, 8)


def test_crop_bbox_clamps_to_image_bounds() -> None:
    image = Image.new("RGB", (20, 20), color="white")

    crop = crop_bbox(image, (18, 19, 10, 10))

    assert crop.size == (2, 1)
