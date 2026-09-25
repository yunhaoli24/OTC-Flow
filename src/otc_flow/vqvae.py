"""VQ-VAE model wrapped as HuggingFace PreTrainedModel."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from monai.networks.nets.vqvae import VQVAE
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.generic import ModelOutput

if TYPE_CHECKING:
    from collections.abc import Sequence


class OTCVQVAEConfig(PretrainedConfig):
    """Configuration for VQ-VAE model."""

    model_type = "vqvae"

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (96, 96, 192),
        num_res_layers: int = 3,
        num_res_channels: Sequence[int] | int = (96, 96, 192),
        downsample_parameters: Sequence[tuple[int, int, int, int]] | tuple[int, int, int, int] = (
            (2, 4, 1, 1),
            (2, 4, 1, 1),
            (2, 4, 1, 1),
        ),
        upsample_parameters: Sequence[tuple[int, int, int, int, int]] | tuple[int, int, int, int, int] = (
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
            (2, 4, 1, 1, 0),
        ),
        num_embeddings: int = 16384,
        embedding_dim: int = 8,
        embedding_init: str = "normal",
        commitment_cost: float = 0.25,
        decay: float = 0.5,
        epsilon: float = 1e-5,
        dropout: float = 0.0,
        act: tuple | str | None = "RELU",
        output_act: tuple | str | None = None,
        *,
        ddp_sync: bool = True,
        use_checkpointing: bool = False,
        mse_loss_weight: float = 1.0,
        codebook_loss_weight: float = 1.0,
        **kwargs: object,
    ) -> None:
        """Initialize VQ-VAE configuration."""
        super().__init__(**kwargs)
        self.spatial_dims = int(spatial_dims)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channels = tuple(int(c) for c in channels)
        self.num_res_layers = int(num_res_layers)
        self.num_res_channels = num_res_channels
        self.downsample_parameters = downsample_parameters
        self.upsample_parameters = upsample_parameters
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.embedding_init = str(embedding_init)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.dropout = float(dropout)
        self.act = act
        self.output_act = output_act
        self.ddp_sync = bool(ddp_sync)
        self.use_checkpointing = bool(use_checkpointing)
        self.mse_loss_weight = float(mse_loss_weight)
        self.codebook_loss_weight = float(codebook_loss_weight)


@dataclass
class OTCVQVAEOutput(ModelOutput):
    """Output of VQ-VAE model."""

    loss: torch.Tensor | None = None
    reconstructions: torch.Tensor | None = None
    codebook_loss: torch.Tensor | None = None
    latents: torch.Tensor | None = None


class OTCVQVAE(PreTrainedModel):
    """VQ-VAE model with HuggingFace integration."""

    config_class = OTCVQVAEConfig
    main_input_name = "pixel_values"

    def __init__(self, config: OTCVQVAEConfig | None = None) -> None:
        """Initialize VQ model with configuration."""
        config = config or OTCVQVAEConfig()
        super().__init__(config)
        self.model = VQVAE(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.channels,
            num_res_layers=config.num_res_layers,
            num_res_channels=config.num_res_channels,
            downsample_parameters=config.downsample_parameters,
            upsample_parameters=config.upsample_parameters,
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            embedding_init=config.embedding_init,
            commitment_cost=config.commitment_cost,
            decay=config.decay,
            epsilon=config.epsilon,
            dropout=config.dropout,
            act=config.act,
            output_act=config.output_act,
            ddp_sync=config.ddp_sync,
            use_checkpointing=config.use_checkpointing,
        )
        self.post_init()

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_dict: bool = True,
        output_latents: bool = False,
        **_: object,
    ) -> OTCVQVAEOutput | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through VQ-VAE model."""
        reconstructions, codebook_loss = self.model(pixel_values)

        loss: torch.Tensor | None = None
        if labels is not None:
            mse = torch.mean((reconstructions - labels) ** 2)
            loss = self.config.mse_loss_weight * mse + self.config.codebook_loss_weight * codebook_loss

        latents = self.encode(pixel_values) if output_latents else None

        if not return_dict:
            return reconstructions, codebook_loss

        return OTCVQVAEOutput(
            loss=loss,
            reconstructions=reconstructions,
            codebook_loss=codebook_loss,
            latents=latents,
        )

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to latent codes."""
        return self.model.encode(pixel_values)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to images."""
        return self.model.decode(latent)

    def get_last_layer(self) -> torch.Tensor:
        """Get last decoder layer weights for adaptive weight."""
        last_block = self.model.decoder.blocks[-1]
        conv = getattr(last_block, "conv", None)
        if conv is None:
            msg = "Decoder last block does not expose convolution weights."
            raise AttributeError(msg)
        return conv.weight


def load_frozen_vqvae(*, vqvae_source: str | Path) -> tuple[OTCVQVAE, OTCVQVAEConfig]:
    """Load a pretrained VQ-VAE and freeze it for OTC-Flow stage two."""
    source_path = Path(vqvae_source)
    with (source_path / "config.json").open(encoding="utf-8") as handle:
        config_data = json.load(handle)
    if config_data.get("model_type", "vqvae") != "vqvae":
        raise ValueError("The checkpoint must be an OTCVQVAE checkpoint with model_type='vqvae'.")
    model = OTCVQVAE.from_pretrained(str(source_path), device_map="cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model.config
