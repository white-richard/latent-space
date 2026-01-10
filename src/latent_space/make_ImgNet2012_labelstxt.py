from pathlib import Path
import torch

def make_ImgNet2012_labels_txt(imagenet_data_dir: str) -> None:
    """
    Generate the labels.txt file for ImageNet-2012 dataset.
    
    :param imagenet_data_dir: Path to the ImageNet-2012 data directory containing meta.bin
    :type imagenet_data_dir: str
    """
    data_dir = Path(imagenet_data_dir)
    meta_bin_path = data_dir / "meta.bin"

    if not meta_bin_path.exists():
        raise FileNotFoundError(f"meta.bin not found at {meta_bin_path}")
    meta = torch.load(meta_bin_path, map_location="cpu")

    wnid_to_names = meta[0]   # dict: wnid -> (name, scientific_name) or similar
    ordered_wnids = meta[1]   # list: wnids in class-id order

    save_path = data_dir / "labels.txt"

    with open(save_path, "w") as f:
        for class_id, wnid in enumerate(ordered_wnids):
            names = wnid_to_names[wnid]

            # names is like ('kit fox', 'Vulpes macrotis')
            # turn it into a single class_name string
            if isinstance(names, (tuple, list)):
                class_name = ", ".join([n for n in names if n])
            else:
                class_name = str(names)

            f.write(f"{class_id},{class_name}\n")
    print(f"Saved ImageNet-2012 labels to {save_path}")