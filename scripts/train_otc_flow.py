"""Train OTC-Flow stage two from a CT/TEE JSONL manifest."""

from dataclasses import dataclass

from transformers import HfArgumentParser

from otc_flow.train import OTCFlowTrainConfig, train_otc_flow


@dataclass
class Arguments:
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


def main() -> None:
    args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
    train_otc_flow(OTCFlowTrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
