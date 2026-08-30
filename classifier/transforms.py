"""Conservative microscopy transforms for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MultiViewTransform:
    """Create independent augmented views of the same source crop."""

    transform: Any
    views: int = 2

    def __post_init__(self) -> None:
        if self.views < 2:
            raise ValueError("MultiViewTransform requires at least two views")

    def __call__(self, image: Any) -> tuple[Any, ...]:
        return tuple(self.transform(image) for _ in range(self.views))


def build_transform(config: Mapping[str, Any], *, training: bool) -> Any:
    color_mode = str(config.get("color_mode", "rgb")).strip().lower()
    if color_mode not in {"rgb", "grayscale"}:
        raise ValueError("transforms.color_mode must be 'rgb' or 'grayscale'")
    augmentation = config.get("augmentation", {}) if training else {}
    saturation = float(augmentation.get("saturation", 0.08))
    hue = float(augmentation.get("hue", 0.02))
    if training and color_mode == "grayscale" and (saturation != 0 or hue != 0):
        raise ValueError("Grayscale training requires saturation=0 and hue=0")
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for image transforms") from exc

    size = int(config.get("image_size", 224))
    operations: list[Any] = []
    if color_mode == "grayscale":
        # Keep three channels because pretrained DINO backbones expect RGB-shaped input.
        operations.append(transforms.Grayscale(num_output_channels=3))
    if training:
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=tuple(augmentation.get("crop_scale", [0.85, 1.0])),
                    ratio=tuple(augmentation.get("crop_ratio", [0.9, 1.1])),
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(float(augmentation.get("horizontal_flip", 0.5))),
                transforms.RandomVerticalFlip(float(augmentation.get("vertical_flip", 0.5))),
            ]
        )
        rotation = float(augmentation.get("rotation_degrees", 180))
        if rotation > 0:
            # Reflection padding prevents black triangular corners from becoming
            # an artificial shortcut after arbitrary rotations.
            padding = max(1, int(round(size * 0.22)))
            operations.extend(
                [
                    transforms.Pad(padding, padding_mode="reflect"),
                    transforms.RandomRotation(rotation),
                    transforms.CenterCrop(size),
                ]
            )
        operations.append(
            transforms.ColorJitter(
                    brightness=float(augmentation.get("brightness", 0.15)),
                    contrast=float(augmentation.get("contrast", 0.15)),
                    saturation=saturation,
                    hue=hue,
            )
        )
        blur_probability = float(augmentation.get("blur_probability", 0.15))
        if blur_probability > 0:
            operations.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=blur_probability,
                )
            )
    else:
        operations.extend([transforms.Resize(size + 32, antialias=True), transforms.CenterCrop(size)])
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    transform = transforms.Compose(operations)
    if training:
        views = int(config.get("augmentation", {}).get("views", 2))
        if views > 1:
            return MultiViewTransform(transform, views=views)
    return transform
