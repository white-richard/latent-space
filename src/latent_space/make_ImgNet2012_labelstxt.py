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

    wnid_to_names = meta[0]  # dict: wnid -> tuple/list of names
    ordered_wnids = meta[1]  # list: wnids in class order

    save_labels_path = data_dir / "labels.txt"

    with open(save_labels_path, "w") as f:
        for wnid in ordered_wnids:
            names = wnid_to_names[wnid]
            main_name = names[0] if isinstance(names, (tuple, list)) else str(names)
            f.write(f"{wnid},{main_name}\n")

    print(f"Saved ImageNet-2012 labels to {save_labels_path}")