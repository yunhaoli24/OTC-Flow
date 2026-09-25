"""OTC-Flow for unpaired CT-to-TEE generation using flow matching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from flow_matching.path import CondOTProbPath
from torch import nn
from torch.nn import functional
from monai.networks.nets import SegResNet, UNet
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.generic import ModelOutput

from .vqvae import OTCVQVAE, OTCVQVAEConfig, load_frozen_vqvae

def _get_group_count(num_channels: int, max_groups: int = 8) -> int:
    max_groups = min(int(max_groups), int(num_channels))
    for groups in range(max_groups, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class CTConditionEncoder(nn.Module):
    """SegResNet CT encoder used by the published OTC-Flow checkpoint."""

    def __init__(
        self,
        *,
        in_channels: int,
        condition_dim: int,
        downsample_strides: tuple[int, ...],
        hidden_channels: tuple[int, ...],
        out_channels: int,
    ) -> None:
        """Initialize the SegResNet condition encoder."""
        super().__init__()
        self.condition_dim = int(condition_dim)
        if any(int(s) != 2 for s in downsample_strides):
            raise ValueError("SegResNet requires VQ-VAE downsample strides of 2.")
        init_filters = int(hidden_channels[0]) if hidden_channels else 32
        num_downsamples = len(downsample_strides)
        self.backbone = SegResNet(
            spatial_dims=2,
            init_filters=init_filters,
            in_channels=int(in_channels),
            out_channels=1,
            blocks_down=(1,) + (2,) * num_downsamples,
            blocks_up=(1,) * num_downsamples,
            use_conv_final=False,
        )
        self.to_out = nn.Conv2d(int(init_filters * (2**num_downsamples)), int(out_channels), kernel_size=1)

    def forward(self, ct: torch.Tensor, _conditions: torch.Tensor) -> torch.Tensor:
        """Encode a CT tensor to the VQ-VAE latent spatial resolution."""
        x, _ = self.backbone.encode(ct)
        return self.to_out(x)


class SegmentationHead(nn.Module):
    """Segmentation head for generating anatomical structure masks.

    Uses a UNet architecture to produce segmentation logits from latent
    representations, enabling training with anatomical constraints.

    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, out_channels: int = 1) -> None:
        """Initialize segmentation head.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Base number of hidden channels.
            out_channels: Number of output channels (segmentation classes).

        """
        super().__init__()
        hidden_channels = int(hidden_channels)
        self.net = UNet(
            spatial_dims=2,
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            channels=(
                hidden_channels,
                hidden_channels * 2,
                hidden_channels * 4,
                hidden_channels * 8,
            ),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=("INSTANCE", {"affine": True}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of segmentation head.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Segmentation logits of shape (B, out_channels, H, W).

        """
        return self.net(x)


class LatentVelocityNet(nn.Module):
    """CNN velocity field used by the published OTC-Flow checkpoint."""

    def __init__(
        self,
        channels: int,
        condition_channels: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        channels = int(channels)
        condition_channels = int(condition_channels)
        hidden_channels = int(hidden_channels)
        num_blocks = int(num_blocks)
        num_groups = _get_group_count(hidden_channels, 8)
        self.in_proj = nn.Conv2d(channels + 1 + condition_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            )
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels)
        self.out_scale = 0.1
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass of velocity network.

        Args:
            x: Latent tensor of shape (B, C, H, W).
            t: Time/timestep tensor of shape (B,).
            condition: Condition tensor of shape (B, C', H, W).

        Returns:
            Velocity prediction of shape (B, C, H, W).

        """
        t_map = t[:, None, None, None].to(dtype=x.dtype).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        h = self.in_proj(torch.cat([x, t_map, condition], dim=1))
        for block in self.blocks:
            h = h + block(h)
        h = self.out_proj(functional.silu(self.out_norm(h)))
        return h * float(self.out_scale)


def _sinkhorn_log_plan(
    cost: torch.Tensor,
    *,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    epsilon: float,
    num_iters: int,
) -> torch.Tensor:
    if cost.ndim != 2:
        msg = f"Expected cost shape (B, Q), got {tuple(cost.shape)}"
        raise ValueError(msg)
    if log_a.ndim != 1 or log_a.shape[0] != cost.shape[0]:
        msg = f"Expected log_a shape ({cost.shape[0]},), got {tuple(log_a.shape)}"
        raise ValueError(msg)
    if log_b.ndim != 1 or log_b.shape[0] != cost.shape[1]:
        msg = f"Expected log_b shape ({cost.shape[1]},), got {tuple(log_b.shape)}"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = f"Expected epsilon > 0, got {epsilon}"
        raise ValueError(msg)
    num_iters = int(num_iters)
    if num_iters < 1:
        msg = f"Expected num_iters >= 1, got {num_iters}"
        raise ValueError(msg)

    log_k = -cost / float(epsilon)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(log_k + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_k + u[:, None], dim=0)
    return (log_k + u[:, None] + v[None, :]).exp()


class OTCFlowConfig(PretrainedConfig):
    """Configuration for OTC-Flow cross-modal generation.

    Contains all hyperparameters for the VQ-VAE encoder, flow-based
    velocity network, and segmentation head.

    """

    model_type = "otc_flow"

    def __init__(
        self,
        *,
        vqvae_source: str = "datasets/vqvae_8w",
        vqvae_config: dict[str, object] | None = None,
        ct_in_channels: int = 1,
        condition_dim: int = 9,
        seg_hidden_channels: int = 32,
        seg_num_classes: int = 8,
        seg_loss_weight: float = 1.0,
        align_loss_weight: float = 0.1,
        encoder_hidden_channels: tuple[int, ...] = (64, 96, 128),
        **kwargs: object,
    ) -> None:
        """Initialize CT-TEE configuration.

        Args:
            vqvae_source: Path to pre-trained VQ-VAE model.
            ct_in_channels: Number of input channels for CT images.
            condition_dim: Dimension of condition embeddings.
            seg_hidden_channels: Base number of hidden channels in segmentation head.
            seg_num_classes: Number of segmentation classes.
            seg_loss_weight: Weight for segmentation loss.
            align_loss_weight: Weight for latent alignment loss.
            encoder_hidden_channels: Hidden channel sizes for CT encoder.
            **kwargs: Additional keyword arguments passed to parent class.

        """
        super().__init__(**kwargs)
        self.vqvae_source = str(vqvae_source)
        self.vqvae_config = vqvae_config
        self.ct_in_channels = int(ct_in_channels)
        self.condition_dim = int(condition_dim)
        self.seg_hidden_channels = int(seg_hidden_channels)
        self.seg_num_classes = int(seg_num_classes)
        self.seg_loss_weight = float(seg_loss_weight)
        self.align_loss_weight = float(align_loss_weight)
        self.encoder_hidden_channels = tuple(int(v) for v in encoder_hidden_channels)


@dataclass
class OTCFlowOutput(ModelOutput):
    """Output of the CT-TEE model for unpaired cross-modal generation.

    Contains segmentation logits, generated TEE images, KID features, latent
    representations, and flow matching outputs.

    """

    logits: torch.Tensor | None = None
    generated_tee: torch.Tensor | None = None
    kid_real_features: torch.Tensor | None = None
    kid_fake_features: torch.Tensor | None = None
    latent_flowed: torch.Tensor | None = None
    latent_q: torch.Tensor | None = None
    latent_reencoded: torch.Tensor | None = None
    fm_vel_pred: torch.Tensor | None = None
    fm_dx_t: torch.Tensor | None = None


class OTCFlowModel(PreTrainedModel):
    """CT-to-TEE generation model with segmentation head using Flow Matching.

    Combines a frozen VQVAE for TEE latent representation, a CT condition encoder,
    OT-based latent flow matching for cross-modal transport, and a segmentation head.
    """

    config_class = OTCFlowConfig

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "OTCFlowModel":
        model = super().from_pretrained(*args, **kwargs)
        model.freeze_vqvae()
        return model

    def __init__(self, config: OTCFlowConfig | None = None) -> None:
        """Initialize OTC-Flow with its anatomical consistency head.

        Args:
            config: Configuration object containing model hyperparameters.
                If None, uses default OTCFlowConfig.

        """
        config = config or OTCFlowConfig()
        super().__init__(config)

        if config.vqvae_source:
            self.vqvae, vq_cfg = load_frozen_vqvae(vqvae_source=config.vqvae_source)
        else:
            vq_cfg = OTCVQVAEConfig(**(config.vqvae_config or {}))
            self.vqvae = OTCVQVAE(vq_cfg)
            self.vqvae.eval()
            for parameter in self.vqvae.parameters():
                parameter.requires_grad_(False)
        self._vq_embedding_dim = int(vq_cfg.embedding_dim)
        self._vq_downsample_strides = tuple(int(p[0]) for p in vq_cfg.downsample_parameters)

        self.fm_path = CondOTProbPath()
        self._fm_weight = 1.0
        self._flow_steps = 5
        self._pair_queue_size = 4096
        self._pair_eps_scale = 0.1
        self._pair_eps_min = 1e-3
        self._pair_num_iters = 50
        self._pair_queue_dtype = torch.float16
        down_factor = 1
        for stride in self._vq_downsample_strides:
            down_factor *= int(stride)
        if 256 % down_factor != 0:
            msg = f"Expected image_size=256 divisible by down_factor={down_factor}"
            raise ValueError(msg)
        latent_hw = 256 // down_factor
        self.ct_encoder = CTConditionEncoder(
            in_channels=config.ct_in_channels,
            condition_dim=config.condition_dim,
            downsample_strides=self._vq_downsample_strides,
            hidden_channels=config.encoder_hidden_channels[: len(self._vq_downsample_strides)],
            out_channels=self._vq_embedding_dim,
        )
        self.latent_flow = LatentVelocityNet(
            self._vq_embedding_dim,
            condition_channels=self._vq_embedding_dim,
            hidden_channels=128,
            num_blocks=4,
        )
        self._pair_queue_shape = (
            int(self._pair_queue_size),
            int(self._vq_embedding_dim),
            int(latent_hw),
            int(latent_hw),
        )
        # NOTE:
        # MoCo-style queues are often registered as buffers, but under DDP the default
        # `broadcast_buffers=True` will synchronize buffers from rank0 to other ranks
        # every forward. This breaks per-rank queue state (ptr/filled) and can lead to
        # hangs that surface as NCCL `BROADCAST Numel=1` timeouts.
        #
        # To make the queue robust regardless of DDP `broadcast_buffers`, keep it as
        # a non-buffer cache. It is (1) not trainable and (2) not meant to be saved.
        self._pair_queue: torch.Tensor | None = None
        self._pair_queue_ptr: int = 0
        self._pair_queue_filled: int = 0
        self.segmenter = SegmentationHead(
            in_channels=1,
            hidden_channels=config.seg_hidden_channels,
            out_channels=config.seg_num_classes,
        )
        self._kid_patch_size = 4
        self.post_init()

    def freeze_vqvae(self) -> None:
        self.vqvae.eval()
        for parameter in self.vqvae.parameters():
            parameter.requires_grad_(False)

    def encode(self, ct_pixel_values: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Encode CT image to TEE latent representation.

        Args:
            ct_pixel_values: CT image tensor of shape (B, C, H, W).
            conditions: Condition tensor of shape (B, C', ...).

        Returns:
            Transported latent tensor of shape (B, embedding_dim, H', W').

        """
        # Get condition from CT
        ct_condition = self.ct_encoder(ct_pixel_values, conditions)
        # Transport from noise using condition
        return self.transport_latent(ct_condition)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to TEE image.

        Args:
            latent: Latent tensor of shape (B, embedding_dim, H, W).

        Returns:
            Decoded TEE image tensor of shape (B, 1, 256, 256).

        """
        latent_q, _ = self.vqvae.model.quantize(latent)
        return self.vqvae.decode(latent_q)

    @torch.no_grad()
    def generate(self, ct_pixel_values: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Generate a quantized TEE image from CT and acquisition conditions."""
        self.eval()
        return self.decode(self.encode(ct_pixel_values, conditions))

    def transport_latent(self, condition: torch.Tensor) -> torch.Tensor:
        steps = int(self._flow_steps)
        if steps < 1:
            return torch.randn_like(condition)
        bsz = int(condition.shape[0])
        # Start from Gaussian noise
        x = torch.randn_like(condition)
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = condition.new_full((bsz,), float(i) * dt)
            x = x + self.latent_flow(x, t, condition) * dt
        return x

    @torch.no_grad()
    def _ensure_pair_queue(self, *, device: torch.device) -> None:
        if (
            self._pair_queue is not None
            and tuple(self._pair_queue.shape) == tuple(self._pair_queue_shape)
            and self._pair_queue.device == device
        ):
            return
        self._pair_queue = torch.zeros(self._pair_queue_shape, device=device, dtype=self._pair_queue_dtype)
        self._pair_queue_ptr = 0
        self._pair_queue_filled = 0

    @torch.no_grad()
    def _enqueue_real_latents(self, latents: torch.Tensor) -> None:
        self._ensure_pair_queue(device=latents.device)

        if latents.ndim != 4:
            msg = f"Expected latents shape (B,C,H,W), got {tuple(latents.shape)}"
            raise ValueError(msg)
        if latents.shape[1:] != self._pair_queue.shape[1:]:
            msg = (
                "Expected latents channels/spatial to match queue, "
                f"got {tuple(latents.shape)} vs {tuple(self._pair_queue.shape)}"
            )
            raise ValueError(msg)
        latents = latents.detach().to(dtype=self._pair_queue.dtype)
        bsz = int(latents.shape[0])
        if bsz < 1:
            return
        queue_size = int(self._pair_queue.shape[0])
        ptr = int(self._pair_queue_ptr)
        end = ptr + bsz
        if end <= queue_size:
            self._pair_queue[ptr:end].copy_(latents)
        else:
            first = queue_size - ptr
            self._pair_queue[ptr:].copy_(latents[:first])
            self._pair_queue[: end - queue_size].copy_(latents[first:])
        self._pair_queue_ptr = int(end % queue_size)
        filled = int(self._pair_queue_filled)
        self._pair_queue_filled = int(min(queue_size, filled + bsz))

    @torch.no_grad()
    def _paired_real_latents(self, x0_key: torch.Tensor) -> torch.Tensor | None:
        if self._pair_queue is None:
            return None
        filled = int(self._pair_queue_filled)
        if filled < 1:
            return None

        bsz = int(x0_key.shape[0])
        if bsz < 1:
            return None
        x1_bank = self._pair_queue[:filled]

        x0_feat = x0_key.float().mean(dim=(2, 3))
        x1_feat = x1_bank.float().mean(dim=(2, 3))
        cost = torch.cdist(x0_feat, x1_feat, p=2).pow(2)

        eps = max(
            float(self._pair_eps_min),
            float(cost.detach().median()) * float(self._pair_eps_scale),
        )
        log_a = cost.new_full((cost.shape[0],), -math.log(float(cost.shape[0])))
        log_b = cost.new_full((cost.shape[1],), -math.log(float(cost.shape[1])))
        plan = _sinkhorn_log_plan(cost, log_a=log_a, log_b=log_b, epsilon=eps, num_iters=self._pair_num_iters)
        x1_idx = plan.argmax(dim=1)
        return x1_bank[x1_idx]

    def _latent_to_patch_features(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            msg = f"Expected latents shape (B,C,H,W), got {tuple(latent.shape)}"
            raise ValueError(msg)
        patch = int(self._kid_patch_size)
        if patch < 1:
            msg = f"Expected patch size >= 1, got {patch}"
            raise ValueError(msg)
        if patch == 1:
            pooled = latent
        else:
            height, width = int(latent.shape[2]), int(latent.shape[3])
            if height % patch == 0 and width % patch == 0:
                pooled = functional.avg_pool2d(latent, kernel_size=patch, stride=patch)
            else:
                pooled = functional.adaptive_avg_pool2d(
                    latent,
                    output_size=(max(1, height // patch), max(1, width // patch)),
                )
        pooled = pooled.permute(0, 2, 3, 1).contiguous()
        return pooled.view(-1, pooled.shape[-1])

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        ct_pixel_values: torch.Tensor,
        conditions: torch.Tensor,
        real_tee: torch.Tensor | None = None,
        _labels: torch.Tensor | None = None,
        _ct_mask: torch.Tensor | None = None,
        *,
        return_dict: bool = True,
        **_: object,
    ) -> OTCFlowOutput | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of CT-TEE model for segmentation training.

        Args:
            ct_pixel_values: CT image tensor of shape (B, C, H, W).
            conditions: Condition tensor of shape (B, C', ...).
            real_tee: Real TEE image tensor of shape (B, 1, H', W'). Required.
            labels: Ground truth segmentation labels (unused in forward).
            ct_mask: CT mask tensor (unused in forward).
            return_dict: Whether to return OTCFlowOutput or tuple.

        Returns:
            OTCFlowOutput containing generated TEE, segmentation logits, flow matching outputs,
            and KID features, or tuple of (generated_tee, seg_logits) if return_dict=False.

        """
        if real_tee is None:
            msg = "real_tee is required; please attach it in the dataloader (build_2d.py)."
            raise ValueError(msg)
        if real_tee.ndim == 3:
            real_tee = real_tee.unsqueeze(0)
        if real_tee.shape[0] != ct_pixel_values.shape[0]:
            msg = f"real_tee batch size mismatch: {int(real_tee.shape[0])} vs {int(ct_pixel_values.shape[0])}."
            raise ValueError(msg)
        real_tee = real_tee.to(device=ct_pixel_values.device, dtype=ct_pixel_values.dtype)

        # 1. Get CT condition (used as condition for flow, and for OT matching)
        ct_condition = self.ct_encoder(ct_pixel_values, conditions)

        with torch.no_grad():
            latent_real = self.vqvae.encode(real_tee)
            # 2. OT Matching: Match CT condition to Real TEE latents
            x1_pair = self._paired_real_latents(ct_condition.detach())

        if self.training:
            self._enqueue_real_latents(latent_real)

        t_fm = torch.rand((ct_condition.shape[0],), device=ct_condition.device, dtype=torch.float32)
        x1_fm = x1_pair if x1_pair is not None else latent_real

        # 3. Start flow from Gaussian Noise (Stationary Source)
        z0_noise = torch.randn_like(ct_condition)

        fm_sample = self.fm_path.sample(x_0=z0_noise, x_1=x1_fm.float(), t=t_fm)

        # 4. Predict velocity with condition
        vel_pred = self.latent_flow(
            fm_sample.x_t.to(dtype=ct_condition.dtype),
            fm_sample.t.to(dtype=ct_condition.dtype),
            condition=ct_condition,
        )

        # 5. Inference / Transport for segmentation branch
        latent_flowed = self.transport_latent(ct_condition)
        latent_q, _ = self.vqvae.model.quantize(latent_flowed)
        generated = self.vqvae.decode(latent_q)

        latent_reencoded = self.vqvae.encode(generated)

        with torch.no_grad():
            kid_real_features = self._latent_to_patch_features(latent_real.float())
            kid_fake_features = self._latent_to_patch_features(latent_reencoded.float())

        seg_logits = self.segmenter(generated)

        if not return_dict:
            return generated, seg_logits

        return OTCFlowOutput(
            logits=seg_logits,
            generated_tee=generated,
            kid_real_features=kid_real_features.detach(),
            kid_fake_features=kid_fake_features.detach(),
            latent_flowed=latent_flowed,
            latent_q=latent_q,
            latent_reencoded=latent_reencoded,
            fm_vel_pred=vel_pred,
            fm_dx_t=fm_sample.dx_t,
        )
