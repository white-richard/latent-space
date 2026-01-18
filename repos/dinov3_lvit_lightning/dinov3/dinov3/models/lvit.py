
from .LViT import (
    DinoVisionTransformer,
    vit_base,
    vit_giant2,
    vit_huge2,
    vit_large,
    vit_so400m,
    vit_7b,
    vit_small,
)


def lvit_small(*args, **kwargs):
    return vit_small(*args, **kwargs)

def lvit_base(*args, **kwargs):
    return vit_base(*args, **kwargs)


def lvit_large(*args, **kwargs):
    return vit_large(*args, **kwargs)


def lvit_huge2(*args, **kwargs):
    return vit_huge2(*args, **kwargs)


def lvit_giant2(*args, **kwargs):
    return vit_giant2(*args, **kwargs)


def lvit_so400m(*args, **kwargs):
    return vit_so400m(*args, **kwargs)


def lvit_7b(*args, **kwargs):
    return vit_7b(*args, **kwargs)


__all__ = [
    "DinoVisionTransformer",
    "lvit_small",
    "lvit_base",
    "lvit_large",
    "lvit_huge2",
    "lvit_giant2",
    "lvit_so400m",
    "lvit_7b",
]
