"""DINOv2 loader with correct frozen/partially-unfrozen gradient handling."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from core.base_model import BaseModel


class DINOBackbone(BaseModel):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14_reg",
        *,
        repository: str = "facebookresearch/dinov2",
        device: str = "cuda",
        unfreeze_layers: int = 0,
        pretrained: bool = True,
    ) -> None:
        self.model_name = model_name
        self.repository = repository
        self.device = self._validated_device(device)
        self.unfreeze_layers = int(unfreeze_layers)
        self.pretrained = bool(pretrained)
        self.model: nn.Module | None = None
        self.embed_dim: int | None = None

    def load(self, source: str | Path | None = None) -> "DINOBackbone":
        if source is not None and Path(str(source)).exists():
            if self.model is None:
                self._load_hub_model()
            state = torch.load(source, map_location="cpu", weights_only=False)
            state_dict = state.get("backbone", state.get("state_dict", state))
            self.model.load_state_dict(state_dict)
        else:
            if source is not None:
                self.model_name = str(source)
            self._load_hub_model()
        self._configure_trainable_layers()
        self.to_device(self.device)
        return self

    def _load_hub_model(self) -> None:
        if self.model_name == "toy_cnn":
            # Offline smoke-test backbone. It exercises the complete optimizer,
            # checkpoint and resume path without downloading DINO weights.
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(16 * 4 * 4, 64),
            )
            self.embed_dim = 64
            return
        self.model = torch.hub.load(
            self.repository,
            self.model_name,
            pretrained=self.pretrained,
        )
        self.embed_dim = int(
            getattr(self.model, "embed_dim", getattr(self.model, "num_features", 0))
        )
        if not self.embed_dim:
            raise RuntimeError("Unable to determine DINO embedding dimension")

    def _configure_trainable_layers(self) -> None:
        if self.model is None:
            raise RuntimeError("Backbone has not been loaded")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if self.unfreeze_layers <= 0:
            self.model.eval()
            return
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Selected DINO model does not expose transformer blocks")
        for block in blocks[-self.unfreeze_layers :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for name in ("norm", "fc_norm"):
            layer = getattr(self.model, name, None)
            if layer is not None:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def to_device(self, device: str | Any) -> "DINOBackbone":
        requested = self._validated_device(device)
        self.device = requested
        if self.model is not None:
            self.model.to(self.device)
        return self

    @staticmethod
    def _validated_device(device: str | Any) -> torch.device:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for DINOv2 but is unavailable. Select the "
                "DiatomDINO GPU Jupyter kernel or explicitly configure device=cpu "
                "for a smoke test."
            )
        return requested

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Backbone has not been loaded")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "embed_dim": self.embed_dim,
                "unfreeze_layers": self.unfreeze_layers,
                "backbone": self.model.state_dict(),
            },
            output,
        )
        return output

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        if self.model is None:
            return iter(())
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def set_train_mode(self, training: bool) -> None:
        if self.model is None:
            raise RuntimeError("Backbone has not been loaded")
        if not training or self.unfreeze_layers <= 0:
            self.model.eval()
            return
        self.model.eval()
        for block in self.model.blocks[-self.unfreeze_layers :]:
            block.train()
        for name in ("norm", "fc_norm"):
            layer = getattr(self.model, name, None)
            if layer is not None:
                layer.train()

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Backbone has not been loaded")
        has_trainable = any(parameter.requires_grad for parameter in self.model.parameters())
        context = nullcontext() if has_trainable and torch.is_grad_enabled() else torch.no_grad()
        with context:
            if hasattr(self.model, "forward_features"):
                output = self.model.forward_features(images)
                if isinstance(output, dict):
                    for key in ("x_norm_clstoken", "x_prenorm", "cls_token"):
                        if key in output:
                            value = output[key]
                            return value[:, 0] if value.ndim == 3 else value
                    raise RuntimeError(f"Unsupported DINO output keys: {sorted(output)}")
                return output[:, 0] if output.ndim == 3 else output
            output = self.model(images)
            return output[:, 0] if output.ndim == 3 else output
