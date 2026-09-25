"""Losses used by the public OTC-Flow Lightning modules."""

from __future__ import annotations

import torch
from torch.nn import functional


def flow_matching_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Conditional flow-matching velocity loss."""
    return functional.mse_loss(predicted.float(), target.float())


def latent_alignment_loss(latent_flowed, latent_q, latent_reencoded, flow_velocity, flow_target, fm_weight):
    """Match transported, quantized and re-encoded latents as in the paper."""
    fm_loss = flow_matching_loss(flow_velocity, flow_target)
    commit_loss = functional.mse_loss(latent_flowed, latent_q.detach())
    cycle_loss = functional.mse_loss(latent_flowed, latent_reencoded.detach())
    return 0.5 * (commit_loss + cycle_loss) + float(fm_weight) * fm_loss, fm_loss


def segmentation_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Dice plus cross-entropy anatomical consistency loss."""
    target = labels.long()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim == 4 and target.shape[1] != 1:
        target = target.argmax(dim=1, keepdim=True)
    one_hot = functional.one_hot(target[:, 0], num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    probabilities = logits.softmax(dim=1)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
    dice = ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    return functional.cross_entropy(logits, target[:, 0]) + (1.0 - dice)


def hinge_generator_loss(logits):
    """Multi-scale hinge generator loss."""
    return torch.stack([-logit.mean() for logit in logits]).mean()


def hinge_discriminator_loss(logits_real, logits_fake):
    """Multi-scale hinge discriminator loss."""
    losses = [
        0.5 * (functional.relu(1.0 - real).mean() + functional.relu(1.0 + fake).mean())
        for real, fake in zip(logits_real, logits_fake, strict=False)
    ]
    return torch.stack(losses).mean()


def feature_matching_loss(features_fake, features_real):
    """Match discriminator features without backpropagating through real images."""
    losses = [
        functional.l1_loss(fake, real.detach())
        for fake_layers, real_layers in zip(features_fake, features_real, strict=False)
        for fake, real in zip(fake_layers, real_layers, strict=False)
    ]
    return torch.stack(losses).mean() if losses else features_fake[0][0].new_zeros(())


class RadialPSD:
    """Radial log power spectral density used for texture statistics."""

    def __init__(self) -> None:
        self._cache = {}

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.shape[1] > 1:
            images = images.mean(dim=1, keepdim=True)
        height, width = images.shape[-2:]
        key = (int(height), int(width), str(images.device))
        if key not in self._cache:
            yy, xx = torch.meshgrid(
                torch.arange(height, device=images.device),
                torch.arange(width, device=images.device),
                indexing="ij",
            )
            radius = torch.sqrt(
                (yy - (height - 1.0) / 2.0) ** 2 + (xx - (width - 1.0) / 2.0) ** 2
            ).round().long().flatten()
            counts = torch.bincount(radius).to(images.dtype).clamp_min(1.0)
            self._cache[key] = radius, counts
        radius, counts = self._cache[key]
        spectrum = torch.fft.fftshift(torch.fft.fft2(images, norm="ortho"), dim=(-2, -1))
        power = (spectrum.real.square() + spectrum.imag.square()).flatten(1)
        radial = images.new_zeros((images.shape[0], counts.numel()))
        radial.scatter_add_(1, radius[None].expand(images.shape[0], -1), power)
        return torch.log1p(radial / counts[None])
