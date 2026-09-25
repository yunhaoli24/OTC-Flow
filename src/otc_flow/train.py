"""PyTorch Lightning training for OTC-Flow stage one and stage two."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator
from torch.utils.data import DataLoader

from .data import ImageDataset, ManifestDataset
from .losses import (
    RadialPSD,
    feature_matching_loss,
    hinge_discriminator_loss,
    hinge_generator_loss,
    latent_alignment_loss,
)
from .model import OTCFlowConfig, OTCFlowModel
from .vqvae import OTCVQVAE, OTCVQVAEConfig


@dataclass
class VQVAETrainConfig:
    image_manifest: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_workers: int = 4
    disc_weight: float = 0.5


@dataclass
class OTCFlowTrainConfig:
    manifest: str
    vqvae_source: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_workers: int = 4
    seg_loss_weight: float = 1.0
    align_loss_weight: float = 1.0
    fm_weight: float = 0.1
    texture_weight: float = 0.05
    disc_weight: float = 0.5
    feature_match_weight: float = 0.0
    disc_start_step: int = 0
    seg_num_classes: int = 8


def _loader(dataset: torch.utils.data.Dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _disc_outputs(value: object):
    logits, features = value
    return logits, features


class VQVAETrainingModule(pl.LightningModule):
    """Adversarial VQ-VAE pretraining module."""

    automatic_optimization = False

    def __init__(self, config: VQVAETrainConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = OTCVQVAE(OTCVQVAEConfig())
        self.discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=3,
            spatial_dims=2,
            channels=64,
            in_channels=1,
            minimum_size_im=256,
        )

    def training_step(self, images: torch.Tensor, _batch_idx: int) -> None:
        generator_optimizer, discriminator_optimizer = self.optimizers()
        reconstruction, codebook_loss = self.model.model(images)
        fake_logits, _ = _disc_outputs(self.discriminator(reconstruction))
        generator_loss = (
            torch.mean(torch.abs(reconstruction - images))
            + codebook_loss
            - self.hparams.config.disc_weight * torch.stack([value.mean() for value in fake_logits]).mean()
        )
        generator_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(generator_loss)
        generator_optimizer.step()

        real_logits, _ = _disc_outputs(self.discriminator(images.detach()))
        fake_logits, _ = _disc_outputs(self.discriminator(reconstruction.detach()))
        discriminator_loss = hinge_discriminator_loss(real_logits, fake_logits)
        discriminator_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()
        self.log("train/generator", generator_loss, prog_bar=True)
        self.log("train/discriminator", discriminator_loss)

    def configure_optimizers(self):
        return (
            torch.optim.AdamW(self.model.parameters(), lr=self.hparams.config.learning_rate),
            torch.optim.AdamW(self.discriminator.parameters(), lr=self.hparams.config.learning_rate),
        )


class OTCFlowTrainingModule(pl.LightningModule):
    """Stage-two Lightning module matching the published OTC-Flow objective."""

    automatic_optimization = False

    def __init__(self, config: OTCFlowTrainConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = OTCFlowModel(
            OTCFlowConfig(
                vqvae_source=config.vqvae_source,
                seg_num_classes=config.seg_num_classes,
                seg_loss_weight=config.seg_loss_weight,
                align_loss_weight=config.align_loss_weight,
            )
        )
        self.discriminator = MultiScalePatchDiscriminator(
            num_d=1,
            num_layers_d=3,
            spatial_dims=2,
            channels=64,
            in_channels=1,
            minimum_size_im=256,
        )
        self.radial_psd = RadialPSD()
        self.segmentation_criterion = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
        )

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> None:
        generator_optimizer, discriminator_optimizer = self.optimizers()
        outputs = self.model(**batch, return_dict=True)
        real_tee = batch["real_tee"]
        labels = batch.get("labels", batch.get("ct_mask"))
        seg = self.segmentation_criterion(outputs.logits, labels)
        align, fm = latent_alignment_loss(
            outputs.latent_flowed,
            outputs.latent_q,
            outputs.latent_reencoded,
            outputs.fm_vel_pred,
            outputs.fm_dx_t,
            self.hparams.config.fm_weight,
        )
        texture = torch.nn.functional.l1_loss(self.radial_psd(outputs.generated_tee), self.radial_psd(real_tee))

        self.discriminator.requires_grad_(False)
        fake_logits, fake_features = _disc_outputs(self.discriminator(outputs.generated_tee))
        real_logits, real_features = _disc_outputs(self.discriminator(real_tee))
        adversarial = hinge_generator_loss(fake_logits)
        feature_matching = feature_matching_loss(fake_features, real_features)
        disc_weight = (
            self.hparams.config.disc_weight
            if self.global_step >= self.hparams.config.disc_start_step
            else 0.0
        )
        generator_loss = (
            self.hparams.config.seg_loss_weight * seg
            + self.hparams.config.align_loss_weight * align
            + self.hparams.config.texture_weight * texture
            + disc_weight * adversarial
            + self.hparams.config.feature_match_weight * feature_matching
        )
        generator_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(generator_loss)
        generator_optimizer.step()

        self.discriminator.requires_grad_(True)
        real_logits, _ = _disc_outputs(self.discriminator(real_tee.detach()))
        fake_logits, _ = _disc_outputs(self.discriminator(outputs.generated_tee.detach()))
        discriminator_loss = disc_weight * hinge_discriminator_loss(real_logits, fake_logits)
        discriminator_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()

        self.log("train/loss", generator_loss, prog_bar=True)
        self.log("train/seg_loss", seg)
        self.log("train/align_loss", align)
        self.log("train/fm_loss", fm)
        self.log("train/texture_loss", texture)
        self.log("train/adv_loss", adversarial)
        self.log("train/discriminator", discriminator_loss)

    def configure_optimizers(self):
        generator = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.hparams.config.learning_rate,
        )
        discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=self.hparams.config.learning_rate)
        return generator, discriminator


def train_vqvae(config: VQVAETrainConfig) -> None:
    """Train and export the stage-one VQ-VAE in Hugging Face format."""
    module = VQVAETrainingModule(config)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir=config.output_dir,
    )
    trainer.fit(module, _loader(ImageDataset(config.image_manifest), config.batch_size, config.num_workers))
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(config.output_dir)


def train_otc_flow(config: OTCFlowTrainConfig) -> None:
    """Train and export stage two in Hugging Face format."""
    module = OTCFlowTrainingModule(config)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir=config.output_dir,
    )
    trainer.fit(module, _loader(ManifestDataset(config.manifest), config.batch_size, config.num_workers))
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(config.output_dir)
