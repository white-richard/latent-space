from nvidia.dali import fn, types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def build_dali_pipeline(
    images_dir: str,
    batch_size: int = 16,
    num_threads: int = 4,
    device_id: int = 0,
    resize_x: int = 256,
    resize_y: int = 256,
    crop_h: int = 224,
    crop_w: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    random_shuffle: bool = True,
) -> DALIGenericIterator:
    """Build and returns a DALI data iterator for image classification training.

    From https://github.com/NVIDIA/DALI

    Args:
        images_dir:     Path to the root directory containing images.
        batch_size:     Number of samples per batch.
        num_threads:    Number of CPU threads used for data loading.
        device_id:      GPU device ID to use for decoding/processing.
        resize_x:       Width to resize images to before cropping.
        resize_y:       Height to resize images to before cropping.
        crop_h:         Height of the final crop.
        crop_w:         Width of the final crop.
        mean:           Per-channel mean for normalization (in [0, 255] scale).
                        Defaults to ImageNet mean.
        std:            Per-channel std for normalization (in [0, 255] scale).
                        Defaults to ImageNet std.
        random_shuffle: Whether to shuffle the dataset.

    Returns:
        DALIGenericIterator yielding dicts with 'data' and 'label' keys.

    """
    if mean is None:
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    if std is None:
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    @pipeline_def(num_threads=num_threads, device_id=device_id)
    def _pipeline():
        images, labels = fn.readers.file(
            file_root=images_dir,
            random_shuffle=random_shuffle,
            name="Reader",
        )
        images = fn.decoders.image_random_crop(
            images,
            device="mixed",
            output_type=types.RGB,
        )
        images = fn.resize(images, resize_x=resize_x, resize_y=resize_y)
        images = fn.crop_mirror_normalize(
            images,
            crop_h=crop_h,
            crop_w=crop_w,
            mean=mean,
            std=std,
            mirror=fn.random.coin_flip(),
        )
        return images, labels

    pipeline = _pipeline(batch_size=batch_size)

    return DALIGenericIterator(
        [pipeline],
        ["data", "label"],
        reader_name="Reader",
    )
